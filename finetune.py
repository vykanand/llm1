import os
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig
import torch

def setup_environment():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def check_model_and_tokenizer(model_name):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token set to EOS token.")
        
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        model = get_peft_model(model, lora_config)
        
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
    except Exception as e:
        print(f"Error checking data format: {e}")
        return None

def preprocess_function(examples, tokenizer, max_length=512):
    # Tokenize questions and answers with consistent padding and truncation
    inputs = tokenizer(examples['question'], max_length=max_length, truncation=True, padding='max_length')
    targets = tokenizer(examples['answer'], max_length=max_length, truncation=True, padding='max_length')
    
    # Ensure consistent length
    model_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    model_inputs['labels'] = torch.tensor(targets['input_ids'])
    
    return model_inputs

def preprocess_and_train_model(tokenizer, model, data, cache_dir, hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            'batch_size': 8,
            'num_epochs': 5,
            'learning_rate': 2e-5,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'logging_steps': 100,
            'save_steps': 5000,
            'eval_steps': 1000
        }

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    # Tokenize and preprocess data
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['question', 'answer'])

    # Split dataset into training and validation sets
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split['train']
    eval_dataset = split['test']

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=hyperparameters.get('batch_size', 8),
        per_device_eval_batch_size=hyperparameters.get('batch_size', 8),
        output_dir=os.path.join(cache_dir, 'results'),
        num_train_epochs=hyperparameters.get('num_epochs', 5),
        learning_rate=hyperparameters.get('learning_rate', 2e-5),
        warmup_steps=hyperparameters.get('warmup_steps', 1000),
        weight_decay=hyperparameters.get('weight_decay', 0.01),
        logging_dir=os.path.join(cache_dir, 'logs'),
        logging_steps=hyperparameters.get('logging_steps', 100),
        save_steps=hyperparameters.get('save_steps', 5000),
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps=hyperparameters.get('eval_steps', 1000),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
        fp16=True,
        remove_unused_columns=False
    )

    # Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
    model.save_pretrained(fine_tuned_model_dir)
    tokenizer.save_pretrained(fine_tuned_model_dir)
    print(f"Fine-tuned model saved to {fine_tuned_model_dir}")

def main():
    model_name = "gpt2"
    dataset_path = '/kaggle/input/tuning/tuning.json'
    cache_dir = setup_environment()
    
    tokenizer, model = check_model_and_tokenizer(model_name)
    
    if tokenizer and model:
        data = check_data_format(dataset_path)
        
        if data:
            preliminary_tuning = input("Do you want to perform preliminary fine-tuning with a subset of the data? (yes/no): ").strip().lower()
            if preliminary_tuning == 'yes':
                print("Performing preliminary fine-tuning.")
                preprocess_and_train_model(tokenizer, model, data[:10], cache_dir)  # Use subset for preliminary tuning
            
            proceed = input("Do you want to proceed with full fine-tuning? (yes/no): ").strip().lower()
            if proceed == 'yes':
                print("Proceeding to full fine-tuning.")
                preprocess_and_train_model(tokenizer, model, data, cache_dir)
            else:
                print("Fine-tuning aborted.")
        else:
            print("Data format is not valid. Aborting.")
    else:
        print("Model or tokenizer failed to load. Aborting.")

if __name__ == '__main__':
    main()
