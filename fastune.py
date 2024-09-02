import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import textstat
from typing import List, Dict, Tuple, Optional

# Utility functions
def get_device():
    """Determine and return the device to use."""
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def ensure_tensor_on_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Ensure tensor is on the correct device."""
    return tensor.to(device)

def check_model_and_tokenizer(model_name: str) -> Tuple[Optional[GPT2Tokenizer], Optional[GPT2LMHeadModel], torch.device]:
    """Load model and tokenizer, and return them along with the device."""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token set to EOS token.")
        
        model = GPT2LMHeadModel.from_pretrained(model_name)
        device = get_device()
        model.to(device)
        
        print(f"Model and tokenizer for '{model_name}' loaded successfully on {device}.")
        return tokenizer, model, device
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None, None

def preprocess_function(examples, tokenizer):
    """Tokenize the input data."""
    return tokenizer(examples['question'], truncation=True, padding='max_length', max_length=512)

def preprocess_and_train_model(tokenizer, model, data, cache_dir, hyperparameters=None):
    """Preprocess data and train the model."""
    if hyperparameters is None:
        hyperparameters = {
            'batch_size': 8,
            'num_epochs': 5,
            'learning_rate': 2e-5,
            'warmup_steps': 1000,
            'weight_decay': 0.01,
            'logging_steps': 100,
            'save_steps': 5000,
            'eval_steps': 1000,
            'gradient_accumulation_steps': 1,
            'fp16_opt_level': 'O1',
            'max_grad_norm': 1.0,
            'adam_epsilon': 1e-8
        }

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=['question', 'answer'])
    
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split['train']
    eval_dataset = split['test']

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
        gradient_accumulation_steps=hyperparameters.get('gradient_accumulation_steps', 1),
        max_grad_norm=hyperparameters.get('max_grad_norm', 1.0),
        adam_epsilon=hyperparameters.get('adam_epsilon', 1e-8),
        fp16_opt_level=hyperparameters.get('fp16_opt_level', 'O1'),
        remove_unused_columns=False,
        save_safetensors=False
    )

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

    try:
        trainer.train()
        fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
        os.makedirs(fine_tuned_model_dir, exist_ok=True)
        model.save_pretrained(fine_tuned_model_dir)
        tokenizer.save_pretrained(fine_tuned_model_dir)
        config = model.config.to_dict()
        with open(os.path.join(fine_tuned_model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Fine-tuned model and tokenizer saved to {fine_tuned_model_dir}")
    except Exception as e:
        print(f"Error during training: {e}")

def generate_answer(prompt, tokenizer, model, max_length=50):
    """Generate an answer from the prompt."""
    device = next(model.parameters()).device  # Get device of the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: ensure_tensor_on_device(value, device) for key, value in inputs.items()}

    try:
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length + len(inputs['input_ids'][0]),
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.85,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""

def compute_similarity(expected_answer: str, generated_answer: str, sentence_model) -> float:
    """Compute the similarity score between the expected and generated answers."""
    # Implement the similarity computation here
    return 0.0  # Placeholder

def check_grammar(text: str) -> int:
    """Check the grammar of the text."""
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

def evaluate_model_performance(data: List[Dict], tokenizer, model, max_length=50) -> List[Dict]:
    """Evaluate model performance on the given data."""
    results = []
    for item in data:
        prompt = item['question']
        expected_answer = item['answer']
        generated_text = generate_answer(prompt, tokenizer, model, max_length)
        
        similarity_score = compute_similarity(expected_answer, generated_text, None)  # Provide the sentence_model
        grammar_errors = check_grammar(generated_text)
        generated_length = len(generated_text.split())
        deviation_from_expected = abs(len(expected_answer.split()) - generated_length)

        results.append({
            'prompt': prompt,
            'expected_answer': expected_answer,
            'generated_text': generated_text,
            'similarity_score': similarity_score,
            'grammar_errors': grammar_errors,
            'deviation_from_expected': deviation_from_expected
        })

    return results

def setup_environment() -> str:
    """Setup environment directories."""
    cache_dir = '/path/to/cache'
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def check_data_format(dataset_path: str) -> Optional[List[Dict]]:
    """Check and load the dataset."""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        if not all('question' in item and 'answer' in item for item in data):
            raise ValueError("Dataset format is invalid.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def generate_from_dataset(data, tokenizer, model):
    """Generate answers from the dataset and print results."""
    results = evaluate_model_performance(data, tokenizer, model)
    for result in results:
        print(f"Prompt: {result['prompt']}")
        print(f"Expected Answer: {result['expected_answer']}")
        print(f"Generated Text: {result['generated_text']}")
        print(f"Similarity Score: {result['similarity_score']}")
        print(f"Grammar Errors: {result['grammar_errors']}")
        print(f"Deviation from Expected: {result['deviation_from_expected']}")
        print("-" * 50)

def main():
    model_name = "gpt2"
    dataset_path = '/kaggle/input/tuning/tuning.json'
    cache_dir = setup_environment()
    
    tokenizer, model, device = check_model_and_tokenizer(model_name)
    
    if tokenizer and model:
        data = check_data_format(dataset_path)
        
        if data:
            print("Choose an option:")
            print("1. Perform preliminary fine-tuning with a subset of the data.")
            print("2. Proceed with full fine-tuning using the entire dataset.")
            print("3. Generate answers using the dataset.")
            print("4. Abort the process.")
            
            choice = input("Enter the number of your choice: ").strip()
            
            if choice == '1':
                print("Performing preliminary fine-tuning.")
                preprocess_and_train_model(tokenizer, model, data[:10], cache_dir)
            elif choice == '2':
                print("Proceeding to full fine-tuning.")
                preprocess_and_train_model(tokenizer, model, data, cache_dir)
            elif choice == '3':
                print("Generating answers from the dataset.")
                generate_from_dataset(data, tokenizer, model)
            elif choice == '4':
                print("Process aborted.")
            else:
                print("Invalid choice. Aborting.")
        else:
            print("Data format is not valid. Aborting.")
    else:
        print("Model or tokenizer failed to load. Aborting.")

if __name__ == '__main__':
    main()
