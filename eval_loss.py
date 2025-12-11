# eval_loss.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = "models/TinyLlama-1.1B-Chat-v1.0"
adapter = "./fine_tuned_grammar_100/checkpoint-100"  # your PEFT
# small eval set: replace with 3-10 representative (prompt, target) pairs from your train/val
pairs = [
    ("### Instruction:\nExplain briefly\n### Input:\nwhat is Faith?\n### Response:\n",
     "Faith is the belief in something that cannot be seen or proven."),
    # add more pairs
]

tokenizer = AutoTokenizer.from_pretrained(base)
m_base = AutoModelForCausalLM.from_pretrained(base).to(device)
m_peft = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), adapter).to(device)

def loss_on_model(model):
    model.eval()
    losses = []
    for prompt, target in pairs:
        full = prompt + target
        tok = tokenizer(full, return_tensors="pt", truncation=True).to(device)
        # shift so labels align with target tokens — this is simplest: use full text and let model compute loss
        with torch.no_grad():
            out = model(**tok, labels=tok["input_ids"])
            losses.append(out.loss.item())
    return sum(losses)/len(losses)

print("Base loss:", loss_on_model(m_base))
print("PEFT loss:", loss_on_model(m_peft))