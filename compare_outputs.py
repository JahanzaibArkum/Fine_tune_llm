from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base = "models/TinyLlama-1.1B-Chat-v1.0"
adapter = "./fine_tuned_grammar_100/checkpoint-100"
merged = "./merged_model"  # if you created it

prompt = "### Instruction:\nExplain briefly\n### Input:\nwhat is Faith?\n### Response:\n"

# Tokenizer from base (or merged if it has tokenizer files)
tokenizer = AutoTokenizer.from_pretrained(base)

# Base model
m_base = AutoModelForCausalLM.from_pretrained(base).to(device)
inpt = tokenizer(prompt, return_tensors="pt").to(device)
out_base = m_base.generate(**inpt, max_new_tokens=120)
print("BASE:", tokenizer.decode(out_base[0], skip_special_tokens=True))

# PEFT-wrapped model (adapter applied)
m_peft = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter).to(device)
out_peft = m_peft.generate(**inpt, max_new_tokens=120)
print("PEFT:", tokenizer.decode(out_peft[0], skip_special_tokens=True))

# merged (if available)
try:
    m_merged = AutoModelForCausalLM.from_pretrained(merged).to(device)
    out_merged = m_merged.generate(**inpt, max_new_tokens=120)
    print("MERGED:", tokenizer.decode(out_merged[0], skip_special_tokens=True))
except Exception as e:
    print("No merged model available or failed to load:", e)