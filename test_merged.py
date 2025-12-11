from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# use base model tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("./merged_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "### Instruction:\nExplain briefly\n### Input:\nwhat is Faith?\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,          # increase this to desired token count
    do_sample=False,             # deterministic; set True for sampling   
    eos_token_id=tokenizer.eos_token_id,
)
try:
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    print("Generation failed:", e)