from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
from scrapper import get_data

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

texts = get_data("diabetes diet")
encodings = [tokenizer(text, truncation=True, padding="max_length", max_length=512) for text in texts]

dataset = Dataset.from_dict({
    'input_ids': [encoding['input_ids'] for encoding in encodings],
    'attention_mask': [encoding['attention_mask'] for encoding in encodings],
    'labels': [encoding['input_ids'] for encoding in encodings]
})

model.config.pad_token_id = tokenizer.eos_token_id

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
