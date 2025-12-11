from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

prompt = """
### Instruction:
Explain the following question in detail.

### Input:
What is Islam?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
