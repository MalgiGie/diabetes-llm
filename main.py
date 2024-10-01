
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer, pipeline
from datasets import Dataset
from scrapper import get_data
import os
import shutil
import torch

def choose_model():
    folders = [f for f in os.listdir("models") if os.path.isdir(os.path.join("models", f))]

    if not folders:
        print("No models found.")
        return None

    for idx, folder in enumerate(folders):
        print(f"{idx + 1}. {folder}")

    while True:
        try:
            selection = input("Enter the model number: ")
            if selection == "\\exit":
                return None
            elif 0 <= int(selection) - 1 < len(folders):
                return folders[int(selection) - 1]
            else:
                print("Invalid number. Try again or enter \"\\exit.\"")
        except ValueError:
            print("That's not a number. Try again or enter \"\\exit.\"")

def create_model(name):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.eos_token_id

    model.save_pretrained(f"models/{name}")
    tokenizer.save_pretrained(f"models/{name}")

def train_model(name, data):
    if data is None:
        print("No data to train.")
        return
    tokenizer = AutoTokenizer.from_pretrained(f"models/{name}")
    model = GPT2LMHeadModel.from_pretrained(f"models/{name}")
    tokenizer.pad_token = tokenizer.eos_token

    encodings = [tokenizer(text, truncation=True, padding="max_length", max_length=512) for text in data]

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

    if os.path.isdir(f"models/{name}"):
        shutil.rmtree(f"models/{name}")
    model.save_pretrained(f"models/{name}")
    tokenizer.save_pretrained(f"models/{name}")

def chatbot(name):
    print(f"\n{name}:")
    while True:
        prompt = input("<You>: ")

        if prompt == '\\exit':
            break
        else:
            generator = pipeline('text-generation', model=f"models/{name}")
            generated_text = generator(prompt, truncation=False, max_length=100, num_return_sequences=1)
            print(f"<Chatbot>: {generated_text[0]['generated_text']}")


def main():

    while True:
        print("\nMenu:")
        print("1 - Create a new model")
        print("2 - Train a model")
        print("3 - Chatbot")
        print("d - Delete a model")
        print("q - Exit")

        choice = input("<You>: ")
        if choice == '1':
            name = input("Enter model name: ")
            if os.path.isdir(f"models/{name}"):
                print(f"{name} already exists")
            else:
                create_model(name)

        elif choice == '2':
            name = choose_model()
            if name is not None:
                prompt = input("Enter prompt: ")
                if os.path.isdir(f"models/{name}"):
                    train_model(name, get_data(prompt))
                else:
                    print(f"{name} does not exist")

        elif choice == '3':
            name = choose_model()
            if name is not None:
                if os.path.isdir(f"models/{name}"):
                    chatbot(name)
                else:
                    print(f"{name} does not exist")

        elif choice == 'd':
            name = choose_model()
            if name is not None:
                if os.path.isdir(f"models/{name}"):
                    shutil.rmtree(f"models/{name}")
                    print(f"{name} deleted")
                else:
                    print(f"{name} does not exist")

        elif choice == 'q':
            break

        else:
            print("Wrong input")


if __name__ == "__main__":
    main()

