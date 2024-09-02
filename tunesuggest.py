import os
import json
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from transformers import pipeline

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

def preprocess_function(examples, tokenizer):
    inputs = tokenizer(examples['question'], max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    targets = tokenizer(examples['answer'], max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    
    inputs['input_ids'] = inputs['input_ids'].squeeze(0).tolist()
    targets['input_ids'] = targets['input_ids'].squeeze(0).tolist()

    model_inputs = {k: v for k, v in inputs.items()}
    model_inputs['labels'] = targets['input_ids']
    return model_inputs

def analyze_model_performance(model, tokenizer, data):
    # Create a small sample of data for testing
    test_sample = data[:5]
    test_dataset = Dataset.from_dict({
        'question': [item['question'] for item in test_sample],
        'answer': [item['answer'] for item in test_sample]
    })
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    
    print("Model Performance Analysis:")
    for item in test_sample:
        question = item['question']
        expected_answer = item['answer']
        generated_answer = text_generator(question, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Generated Answer: {generated_answer}")
        print()

def suggest_hyperparameters_and_tokenizer_settings(data):
    # Hypothetical evaluation function to suggest improvements
    num_samples = len(data)
    avg_length = np.mean([len(item['question']) for item in data])
    max_length = max([len(item['question']) for item in data])
    
    # Suggested hyperparameters based on typical scenarios
    suggested_batch_size = 4
    suggested_num_epochs = 5
    suggested_learning_rate = 3e-5
    suggested_max_length = min(max_length, 512)
    
    print(f"Suggested Hyperparameters:")
    print(f"Batch Size: {suggested_batch_size}")
    print(f"Number of Epochs: {suggested_num_epochs}")
    print(f"Learning Rate: {suggested_learning_rate}")
    print(f"Maximum Token Length: {suggested_max_length}")

def main():
    model_name = "gpt2"
    dataset_path = '/kaggle/input/t-small/t-small.json'
    cache_dir = setup_environment()
    
    tokenizer, model = check_model_and_tokenizer(model_name)
    
    if tokenizer and model:
        data = check_data_format(dataset_path)
        
        if data:
            analyze_model_performance(model, tokenizer, data)
            suggest_hyperparameters_and_tokenizer_settings(data)
        else:
            print("Data format is not valid. Aborting.")
    else:
        print("Model or tokenizer failed to load. Aborting.")

if __name__ == '__main__':
    main()
