from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F

config = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                            config=config,
                                            torch_dtype=torch.bfloat16,
                                            attn_implementation="eager",
                                            device_map="cpu",
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_loss_weight = 0.3  # Weight for attention-based loss
        self.lm_loss_weight = 0.7         # Weight for language modeling loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get model outputs with attention weights
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions  # List of attention weights for each layer
        
        # Extract labels and shift them for causal LM
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate standard language modeling loss
        lm_loss = self.compute_lm_loss(shift_logits, shift_labels)
        
        # Calculate attention-based loss using model's actual output
        attention_loss = self.compute_attention_loss_from_output(attentions, logits, inputs)
        
        # Combine losses

        total_loss = self.lm_loss_weight * lm_loss + self.attention_loss_weight * attention_loss
        
        # Log losses for monitoring (optional)
        if hasattr(self, 'state') and self.state.global_step % 100 == 0:
            print(f"Step {self.state.global_step}: LM Loss: {lm_loss.item():.4f}, "
                  f"Attention Loss: {attention_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
            
            
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_lm_loss(self, logits, labels):
        """Compute standard language modeling loss"""
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    def compute_attention_loss_from_output(self, attentions, logits, inputs):
        """Compute attention-based loss using model's actual generated output"""
        batch_size = logits.size(0)
        seq_len = logits.size(1)
        
        # Initialize attention loss
        attention_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        
        for batch_idx in range(batch_size):
            # Get the model's predicted tokens for this batch item
            predicted_tokens = torch.argmax(logits[batch_idx], dim=-1)  # (seq_len,)
            
            # Find answer response tokens in the model's output
            answer_start_idx = self.find_answer_start_from_output(predicted_tokens, inputs, batch_idx)
            
            if answer_start_idx is not None and answer_start_idx < seq_len - 1:
                # Calculate attention loss for this batch item
                batch_attention_loss = self.calculate_sentence_to_response_attention_loss(
                    attentions, batch_idx, answer_start_idx, predicted_tokens
                )
                attention_loss += batch_attention_loss
        
        return attention_loss / batch_size
    
    def calculate_sentence_to_response_attention_loss(self, attentions, batch_idx, answer_start_idx, predicted_tokens):
        """Calculate attention loss: average attention from each thinking sentence to response"""
        # First, identify sentences in the thinking process (before answer)
        thinking_sentences = self.extract_thinking_sentences(predicted_tokens, answer_start_idx)
        
        if not thinking_sentences:
            return torch.tensor(0.0, device=predicted_tokens.device, dtype=torch.float32)
        
        # Calculate attention from each thinking sentence to the response
        sentence_attention_scores = []
        
        for sentence_start, sentence_end in thinking_sentences:
            # Calculate average attention from this sentence to the response
            sentence_attention = self.calculate_sentence_to_response_attention(
                attentions, batch_idx, sentence_start, sentence_end, answer_start_idx
            )
            sentence_attention_scores.append(sentence_attention)
        
        # Average attention across all thinking sentences
        avg_sentence_attention = torch.stack(sentence_attention_scores).mean()
        
        # Use 1/avg(attention from thinking sentences to response) as loss
        return 1 - avg_sentence_attention

    def extract_thinking_sentences(self, predicted_tokens, answer_start_idx):
        """Extract sentence boundaries in the thinking process (before answer)"""
        # Decode the thinking portion to text
        thinking_tokens = predicted_tokens[:answer_start_idx]
        thinking_text = tokenizer.decode(thinking_tokens, skip_special_tokens=True)
        
        # Use rule-based sentence splitting
        sentences = self.split_text_into_sentences(thinking_text)
        
        # Convert sentence positions back to token positions
        token_sentences = self.convert_sentences_to_token_positions(sentences, thinking_tokens)
        
        return token_sentences
    
    def split_text_into_sentences(self, text):
        """Split text into sentences using rule-based approach"""
        import re
        
        # Clean the text
        text = text.strip()
        if not text:
            return []
        
        # Try NLTK first (more robust)
        try:
            import nltk
            # Download punkt tokenizer if not available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
            
            if sentences:
                return sentences
                
        except ImportError:
            # NLTK not available, fall back to regex
            pass
        
        # Fallback: Regex-based sentence splitting
        # Split on sentence endings with lookahead to handle abbreviations
        # This regex handles common cases like "Mr.", "Dr.", "U.S.A.", etc.
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split the text
        raw_sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Filter out very short fragments
                sentences.append(sentence)
        
        # Handle edge cases where regex might not catch everything
        if not sentences and text:
            # Final fallback: split on major punctuation
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 3]
        
        return sentences
    
    def convert_sentences_to_token_positions(self, sentences, tokens):
        """Convert sentence text positions to token positions"""
        if not sentences:
            return []
        
        token_sentences = []
        current_pos = 0
        
        for sentence in sentences:
            # Find where this sentence starts in the token sequence
            sentence_start = self.find_sentence_start_in_tokens(sentence, tokens, current_pos)
            if sentence_start is not None:
                # Find where this sentence ends
                sentence_end = self.find_sentence_end_in_tokens(sentence, tokens, sentence_start)
                if sentence_end is not None:
                    token_sentences.append((sentence_start, sentence_end))
                    current_pos = sentence_end
        
        return token_sentences
    
    def find_sentence_start_in_tokens(self, sentence, tokens, start_pos):
        """Find the starting position of a sentence in the token sequence"""
        # Simple approach: find the first occurrence of the sentence start
        sentence_start_words = sentence.split()[:3]  # First few words
        
        for i in range(start_pos, len(tokens)):
            # Check if we can find the sentence start around this position
            context_tokens = tokens[max(0, i-2):i+3]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            
            if any(word.lower() in context_text.lower() for word in sentence_start_words):
                return i
        
        return start_pos  # Fallback to current position
    
    def find_sentence_end_in_tokens(self, sentence, tokens, start_pos):
        """Find the ending position of a sentence in the token sequence"""
        # Look for sentence ending punctuation after start_pos
        for i in range(start_pos, len(tokens)):
            token_text = tokenizer.decode([tokens[i]], skip_special_tokens=True)
            if token_text in ['.', '!', '?']:
                return i + 1
        
        # If no clear ending, use a reasonable boundary
        return min(start_pos + 50, len(tokens))  # Max 50 tokens per sentence
    
    def calculate_sentence_to_response_attention(self, attentions, batch_idx, sentence_start, sentence_end, answer_start_idx):
        """Calculate attention from a specific sentence to the response"""
        
    
        # Get attention weights for this layer
        layer_attention = attentions[-1][batch_idx]  # (num_heads, seq_len, seq_len)
        
        # Calculate attention FROM the sentence TO the response
        # sentence_attention: attention from sentence tokens to response tokens
        sentence_to_response = layer_attention[-1,answer_start_idx:, sentence_start:sentence_end].mean(dim=-1)  # (num_heads, sentence_length)
        
        # Average across sentence tokens and attention heads
        
        total_attention = sentence_to_response
        
        return total_attention

    def find_answer_start_from_output(self, predicted_tokens, inputs, batch_idx):
        """Find the start index of the answer response in the model's predicted output"""
        # Method 1: Look for thinking process markers in decoded text
        decoded_text = tokenizer.decode(predicted_tokens, skip_special_tokens=True)
        
        # Look for thinking markers
        thinking_markers = ["</think>", "<think>"]
        for marker in thinking_markers:
            if marker in decoded_text:
                marker_pos = decoded_text.find(marker)
                # Find the token position corresponding to this text position
                token_pos = self.find_text_position_in_tokens(decoded_text[:marker_pos], predicted_tokens)
                if token_pos is not None:
                    return token_pos + len(tokenizer.encode(marker, add_special_tokens=False))
        
        # Method 2: Use sentence-based detection
        sentences = self.split_text_into_sentences(decoded_text)
        if len(sentences) > 1:
            # Use the last sentence as the answer
            last_sentence_start = decoded_text.rfind(sentences[-1])
            if last_sentence_start != -1:
                token_pos = self.find_text_position_in_tokens(decoded_text[:last_sentence_start], predicted_tokens)
                if token_pos is not None:
                    return token_pos
        
        # Method 3: Use the last third of the sequence as fallback
        return max(0, len(predicted_tokens) // 3)
    
    def find_text_position_in_tokens(self, text_prefix, tokens):
        """Find the token position corresponding to a text position"""
        # Simple approach: find the token sequence that best matches the text prefix
        for i in range(len(tokens)):
            context_tokens = tokens[:i+1]
            context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
            
            if len(context_text) >= len(text_prefix):
                # Check if this is a good match
                if context_text.endswith(text_prefix) or text_prefix.endswith(context_text):
                    return i
        
        return None

    
def format_data_for_sft(dataset):
    """Convert dataset with question/answer columns to prompt/completion format"""
    formatted_data = []
    
    for item in dataset:
        # Transform answer format from <answer></answer> to \boxed{}
        answer_content = item["answer"]
        if "<answer>" in answer_content and "</answer>" in answer_content:
            # Extract content between <answer> tags and wrap with \boxed{}
            start_tag = "<answer>"
            end_tag = "</answer>"
            start_idx = answer_content.find(start_tag) + len(start_tag)
            end_idx = answer_content.find(end_tag)
            if start_idx != -1 and end_idx != -1:
                answer_text = answer_content[start_idx:end_idx].strip()
                # Replace the entire <answer></answer> section with \boxed{}
                answer_content = answer_content.replace(
                    f"{start_tag}{answer_text}{end_tag}", 
                    f"\\boxed{{{answer_text}}}"
                )
        
        formatted_item = {
        "prompt": [{"role": "user", "content": item["question"]}],
        "completion": [
            {"role": "assistant", "content": answer_content}
        ],
    }
        formatted_data.append(formatted_item)
    
    return formatted_data

# Load both train and test splits
train_dataset = load_dataset("Jax-dan/Lite-Thinking", split="train")
test_dataset = load_dataset("Jax-dan/Lite-Thinking", split="test")

# Format the data
formatted_train_data = format_data_for_sft(train_dataset)
formatted_test_data = format_data_for_sft(test_dataset)

# Save formatted data to JSON files
with open("formatted_train_data.json", "w") as f:
    json.dump(formatted_train_data, f, indent=2)

with open("formatted_test_data.json", "w") as f:
    json.dump(formatted_test_data, f, indent=2)

print(f"Train data: {len(formatted_train_data)} samples")
print(f"Test data: {len(formatted_test_data)} samples")
print("Data saved to formatted_train_data.json and formatted_test_data.json")

# Example of formatted data structure
if formatted_train_data:
    print("\nExample formatted data:")
    print(json.dumps(formatted_train_data[0], indent=2))

# Convert back to Hugging Face Dataset objects for SFTTrainer
train_dataset_formatted = Dataset.from_list(formatted_train_data)
test_dataset_formatted = Dataset.from_list(formatted_test_data)

# Use formatted data for training
trainer = CustomSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset_formatted,
    eval_dataset=test_dataset_formatted,
    args=SFTConfig(
        output_dir="checkpoints/attention",
        do_train=True,
        do_eval=True,
        max_steps=5000,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
    ),
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
)

# Configure attention loss weights
trainer.attention_loss_weight = 0.4  # Adjust this value to control attention loss influence
trainer.lm_loss_weight = 0.6         # Standard language modeling loss weight

trainer.train()