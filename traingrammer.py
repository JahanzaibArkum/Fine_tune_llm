from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# -------------------------
# Load dataset (JSONL) and select only 100 examples
# -------------------------
dataset_dict = load_dataset("json", data_files="hadith_grammar.jsonl")
dataset = dataset_dict["train"].select(range(100))  # only first 100 examples

# -------------------------
# Load base model
# -------------------------
model_name = "models/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------
# Tokenization function
# -------------------------
def tokenize_fn(batch):
    outputs = [str(x) for x in batch.get("output", [""]*len(batch["instruction"]))]
    instructions = [str(x) for x in batch.get("instruction", [""]*len(outputs))]
    inputs = [str(x) for x in batch.get("input", [""]*len(outputs))]

    full_texts = []
    for instr, inp, outp in zip(instructions, inputs, outputs):
        prompt = f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n"
        full_texts.append(prompt + outp)

    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=256  # shorter to save memory
    )
    tokenized["labels"] = [list(x) for x in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    batch_size=10,  # batch of 10 examples for CPU
    remove_columns=dataset.column_names
)

# -------------------------
# LoRA configuration
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -------------------------
# Load model and apply LoRA
# -------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
model = get_peft_model(model, lora_config)

# -------------------------
# Training arguments
# -------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_grammar_100",
    per_device_train_batch_size=1,  # CPU
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=5,
    learning_rate=2e-4,
    fp16=False,  # CPU only
    report_to=None
)

# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start training
if __name__ == "__main__":
    trainer.train()
