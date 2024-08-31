from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import os

# Define cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Load tokenizer and model from the cache
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)

# Check and set pad_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load and preprocess the dataset
dataset = load_dataset('json', data_files={'data': 'tuning.json'})

def preprocess_function(examples):
    inputs = examples['question']
    targets = examples['answer']
    
    # Tokenize the inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')
    
    # Add labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Map the preprocessing function to the dataset
tokenized_dataset = dataset['data'].map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    output_dir=os.path.join(cache_dir, 'results'),  # Save results to cache directory
    num_train_epochs=3,
    logging_dir=os.path.join(cache_dir, 'logs'),
    logging_steps=10,
    save_steps=10_000,
    save_total_limit=2,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,  # Use the entire tokenized dataset for training
    eval_dataset=None,                # No evaluation dataset provided
)

# Train the model
trainer.train()

# Save the fine-tuned model to the cache directory
fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
model.save_pretrained(fine_tuned_model_dir)
tokenizer.save_pretrained(fine_tuned_model_dir)
