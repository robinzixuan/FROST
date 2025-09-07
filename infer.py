# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from Qwen_attention import Qwen3AttentionExtrea
from Qwen2_5_attention import Qwen2AttentionExtra
tokenizer = AutoTokenizer.from_pretrained("/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/easy_r1/qwen25_softmax1_math/global_step_5/actor/huggingface")
model = AutoModelForCausalLM.from_pretrained("/projects/p32013/reasoning/AlphaOne/eval/GARPO1/checkpoints/easy_r1/qwen25_softmax1_math/global_step_5/actor/huggingface")

for layer_idx in range(len(model.model.layers)):
    old_attn = model.model.layers[layer_idx].self_attn
    new_attn = Qwen2AttentionExtra(
        config=model.config,
        layer_idx=layer_idx,
        softmax_fn='vanilla'
    )
    new_attn.load_state_dict(old_attn.state_dict(), strict=False)
    model.model.layers[layer_idx].self_attn = new_attn

# We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
# on a small vocab and want a smaller embedding size, remove this test.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

messages = [
    {"role": "user", "content": "There are 9 boys and 12 girls in a class. The teacher needs to create groups with three members for their class activity. How many groups are formed?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))