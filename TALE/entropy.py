import json
import torch
import torch.nn.functional as F
import nltk
from nltk.tokenize import sent_tokenize
from typing import Union
from phi3_attention import Phi3AttentionExtra

nltk.download("punkt")

def compute_entropy(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    Calculate entropy of the prediction distribution based on logits.
    """
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]

    return entropy.item() if is_single_input else entropy

def sentence_entropy_from_logits(sentence_logits: torch.Tensor) -> float:
    """
    Compute sentence entropy as the average token entropy across its tokens.
    
    Args:
        sentence_logits: Tensor of shape [num_tokens, vocab_size]
                         logits for each token in the sentence.
    """
    entropies = []
    for tok_logits in sentence_logits:  # iterate per token
        ent = compute_entropy(tok_logits)
        entropies.append(ent)
    if not entropies:
        return 0.0
    return sum(entropies) / len(entropies)

def average_sentence_entropy(jsonl_path: str, model, tokenizer):
    """
    Read jsonl, extract prediction text, compute avg sentence entropy
    using model token logits.
    """
    all_entropies = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            prediction = entry.get("prediction", "")
            sentences = sent_tokenize(prediction)

            for sent in sentences:
                # Tokenize and get logits
                inputs = tokenizer(sent, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=False)
                    # logits shape: [1, seq_len, vocab_size]
                    sent_logits = outputs.logits.squeeze(0)[:-1]  # ignore last token
                ent = sentence_entropy_from_logits(sent_logits)
                all_entropies.append(ent)

    if not all_entropies:
        return 0.0
    return sum(all_entropies) / len(all_entropies)

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/phi4_vanilla/final"  # you can replace with your model
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager",)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for layer_idx in range(len(model.model.layers)):
        old_attn = model.model.layers[layer_idx].self_attn
        new_attn = Phi3AttentionExtra(
            config=model.config,
            layer_idx=layer_idx,
            softmax_fn='vanilla'
        )
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        model.model.layers[layer_idx].self_attn = new_attn
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    path = "/projects/p32013/reasoning/AlphaOne/eval/TALE/results/softmax/Phi4/AIME/output_with_reasoning.jsonl"  # replace with your file
    avg_entropy = average_sentence_entropy(path, model, tokenizer)
    print(f"Average Sentence Entropy (token-based): {avg_entropy:.4f}")
