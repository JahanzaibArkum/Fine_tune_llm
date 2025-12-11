from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "models/TinyLlama-1.1B-Chat-v1.0"
adapter_checkpoint = "./fine_tuned_grammar_100/checkpoint-100"  # path that contains adapter_config.json & adapter_model.safetensors
out_dir = "./merged_model"

print("Loading base model (this may use a lot of RAM)...")
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

print("Wrapping with PEFT adapter from:", adapter_checkpoint)
model = PeftModel.from_pretrained(model, adapter_checkpoint, torch_dtype=torch.float32)

print("Merging adapter into base weights (merge_and_unload). This may require additional RAM.")
merged = model.merge_and_unload()

print("Saving merged model to:", out_dir)
merged.save_pretrained(out_dir)
print("Done. Merged model saved to", out_dir)
