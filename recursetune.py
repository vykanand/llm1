import os
import json
import pandas as pd
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util
import textstat
import language_tool_python


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
    inputs = tokenizer(examples['question'], max_length=max_length, truncation=True, padding='max_length')
    targets = tokenizer(examples['answer'], max_length=max_length, truncation=True, padding='max_length')
    
    model_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    model_inputs['labels'] = torch.tensor(targets['input_ids'])
    
    return model_inputs

def generate_answer(prompt, tokenizer, model, max_length=100, temperature=0.7, top_p=0.85, top_k=50, no_repeat_ngram_size=2, repetition_penalty=1.2):
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move input tensors to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate output
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length + len(inputs['input_ids'][0]),
        num_return_sequences=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

def compute_similarity(text1, text2, model):
    sentences = [text1, text2]
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def check_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)

def evaluate_model_performance(data, tokenizer, model, max_length=50):
    results = []
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for i, item in enumerate(data):
        prompt = item['question']
        expected_answer = item['answer']
        print(f"Processing Element {i+1}:")
        print(f"Prompt: {prompt}")
        
        generated_text = generate_answer(prompt, tokenizer, model, max_length)
        similarity_score = compute_similarity(expected_answer, generated_text, sentence_model)
        grammar_errors = check_grammar(generated_text)
        generated_length = len(generated_text.split())
        deviation_from_expected = abs(len(expected_answer.split()) - generated_length)
        
        results.append({
            'prompt': prompt,
            'expected_answer': expected_answer,
            'generated_text': generated_text,
            'similarity_score': similarity_score,
            'grammar_errors': grammar_errors,
            'generated_length': generated_length,
            'deviation_from_expected': deviation_from_expected
        })
        
        print(f"Generated Text: {generated_text}")
        print(f"Similarity Score: {similarity_score:.2f}")
        print(f"Number of Grammar Errors: {grammar_errors}")
        print(f"Length of Generated Text: {generated_length} words")
        print(f"Deviation from Expected Answer Length: {deviation_from_expected} words")
        print("="*50)
    
    return results

def print_evaluation_results(results):
    for result in results:
        print("\n" + "="*50)
        print(f"Prompt:\n{result['prompt']}\n")
        print(f"Expected Answer:\n{result['expected_answer']}\n")
        print(f"Generated Text:\n{result['generated_text']}\n")
        print(f"Similarity Score: {result['similarity_score']:.2f}")
        print(f"Number of Grammar Errors: {result['grammar_errors']}")
        print(f"Length of Generated Text: {result['generated_length']} words")
        print(f"Deviation from Expected Answer Length: {result['deviation_from_expected']} words")
        print("="*50)

def preprocess_and_train_model(tokenizer, model, data, cache_dir, hyperparameters):
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

    trainer.train()

    fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
    os.makedirs(fine_tuned_model_dir, exist_ok=True)

    model.save_pretrained(fine_tuned_model_dir)
    tokenizer.save_pretrained(fine_tuned_model_dir)

    print(f"Fine-tuned model and tokenizer saved to {fine_tuned_model_dir}")

def dynamic_hyperparameter_tuning(dataset_path, model_name="gpt2", max_similarity=0.80):
    cache_dir = setup_environment()
    tokenizer, model = check_model_and_tokenizer(model_name)
    
    if tokenizer and model:
        data = check_data_format(dataset_path)
        
        if data:
            best_hyperparameters = {
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
            
            current_similarity = 0.0
            iteration = 0
            
            while current_similarity < max_similarity:
                print(f"\nIteration {iteration+1}:")
                preprocess_and_train_model(tokenizer, model, data[:5], cache_dir, best_hyperparameters)
                
                results = evaluate_model_performance(data, tokenizer, model)
                similarities = [res['similarity_score'] for res in results]
                current_similarity = np.mean(similarities)
                
                print(f"Current Average Similarity Score: {current_similarity:.2f}")

                if current_similarity >= max_similarity:
                    print("Desired similarity achieved.")
                    break
                
                # Update hyperparameters dynamically
                best_hyperparameters['learning_rate'] = best_hyperparameters['learning_rate'] * np.random.uniform(0.8, 1.2)
                best_hyperparameters['num_epochs'] = int(best_hyperparameters['num_epochs'] * np.random.uniform(0.8, 1.2))
                
                iteration += 1
        else:
            print("Data format is not valid. Aborting.")
    else:
        print("Model or tokenizer failed to load. Aborting.")

if __name__ == '__main__':
    dataset_path = '/kaggle/input/tuning/tuning.json'
    dynamic_hyperparameter_tuning(dataset_path)
