from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Base model
base_model = "models/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

# Load LoRA/PEFT weights from the checkpoint directory (point to the folder that contains adapter files)
# Here we point to the checkpoint folder which contains adapter_config.json and adapter_model.safetensors
adapter_path = "./fine_tuned_grammar_100/checkpoint-100"
model = PeftModel.from_pretrained(model, adapter_path)

# Move model to available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use same prompt template used during training so the adapter sees familiar input shape
instr = "Explain the concept briefly"
inp = ""
question = "what is islam?"
prompt = f"### Instruction:\n{instr}\n### Input:\n{question}\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate only new tokens (set max_new_tokens) and avoid returning the entire prompt twice
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
