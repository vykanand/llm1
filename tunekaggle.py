import os
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from datasets import Dataset

def setup_environment():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def check_model_and_tokenizer(model_name):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Set pad_token_id if it is not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Pad token set to EOS token.")
        
        print(f"Model and tokenizer for '{model_name}' loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def check_data_format(dataset_path):
    try:
        with open(dataset_path, 'r') as file:
            data = json.load(file)
        
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError("Data should be a list of dictionaries.")
        
        required_fields = {'question', 'answer'}
        for item in data:
            if not required_fields.issubset(item.keys()):
                raise ValueError(f"Missing required fields in item: {item}")

        print("Data format is valid.")
        return data
    except FileNotFoundError:
        print(f"File not found: {dataset_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {dataset_path}")
    except ValueError as e:
        print(f"Data format error: {e}")
    return None

def check_tokenization(tokenizer, data):
    try:
        sample = data[0]  # Take the first item as a sample
        inputs = tokenizer(sample['question'], return_tensors="pt", padding=True, truncation=True)
        targets = tokenizer(sample['answer'], return_tensors="pt", padding=True, truncation=True)
        
        print(f"Sample input IDs: {inputs['input_ids']}")
        print(f"Sample attention mask: {inputs['attention_mask']}")
        print(f"Sample target IDs: {targets['input_ids']}")
        return True
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return False

def check_model_config(model):
    try:
        config = model.config
        print(f"Model configuration: {config}")
        
        if not hasattr(config, 'max_position_embeddings'):
            raise ValueError("Model configuration is missing 'max_position_embeddings'.")
        
        print("Model configuration is valid.")
        return True
    except Exception as e:
        print(f"Error checking model configuration: {e}")
        return False

def prepare_and_train_model(tokenizer, model, data, cache_dir, subset_size=None):
    if subset_size:
        data = data[:subset_size]  # Limit to subset_size elements for preliminary tuning
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    def preprocess_function(examples):
        inputs = examples['question']
        targets = examples['answer']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Define training arguments with mixed precision and wandb disabled
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        output_dir=os.path.join(cache_dir, 'results'),
        num_train_epochs=1 if subset_size else 3,
        logging_dir=os.path.join(cache_dir, 'logs'),
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",
        report_to="none",  # Disable wandb and other report tools
        fp16=True  # Enable mixed precision training
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset
    )

    try:
        # Start training
        trainer.train()

        # Save fine-tuned model
        fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
        model.save_pretrained(fine_tuned_model_dir)
        tokenizer.save_pretrained(fine_tuned_model_dir)
        print(f"Fine-tuned model saved to {fine_tuned_model_dir}")
    except Exception as e:
        print(f"Error during training or saving the model: {e}")

def main():
    model_name = "gpt2"
    dataset_path = '/kaggle/input/tuning/tuning.json'
    subset_size = 10  # Define the number of samples for preliminary tuning

    cache_dir = setup_environment()
    tokenizer, model = check_model_and_tokenizer(model_name)
    
    if tokenizer and model:
        data = check_data_format(dataset_path)
        
        if data and check_tokenization(tokenizer, data) and check_model_config(model):
            print("All checks passed.")
            preliminary_tuning = input("Do you want to perform preliminary fine-tuning with a subset of the data? (yes/no): ").strip().lower()
            if preliminary_tuning == 'yes':
                print("Performing preliminary fine-tuning.")
                prepare_and_train_model(tokenizer, model, data, cache_dir, subset_size=subset_size)
            
            proceed = input("Do you want to proceed with full fine-tuning? (yes/no): ").strip().lower()
            if proceed == 'yes':
                print("Proceeding to full fine-tuning.")
                prepare_and_train_model(tokenizer, model, data, cache_dir, subset_size=None)
            else:
                print("Fine-tuning aborted.")
        else:
            print("One or more checks failed. Aborting fine-tuning.")
    else:
        print("Model or tokenizer failed to load. Aborting fine-tuning.")

if __name__ == '__main__':
    main()
