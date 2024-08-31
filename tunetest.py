import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the cache directory and model paths
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_name = "gpt2"
fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)

# Set pad_token_id if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_text(prompt, max_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_tokens + len(inputs['input_ids'][0]),  # Ensure total length does not exceed model limit
        num_return_sequences=1,
        temperature=0.7,  # Adjust temperature for more varied outputs
        top_p=0.9,        # Adjust top_p for more diverse outputs
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    return generated_text

def evaluate_performance(data_file, max_tokens=50):
    # Load data from JSON file
    with open(data_file, 'r') as file:
        data = json.load(file)
    
    results = []
    for item in data:
        prompt = item['question']
        expected_answer = item['answer']
        generated_text = generate_text(prompt, max_tokens)
        results.append({
            'prompt': prompt,
            'expected_answer': expected_answer,
            'generated_text': generated_text,
            'length': len(generated_text.split())  # Length in words
        })
    return results

# Path to the JSON file containing questions and answers
data_file = 'tuning.json'

# Evaluate the performance
performance_results = evaluate_performance(data_file)

# Print the results
for result in performance_results:
    print(f"Prompt: {result['prompt']}")
    print(f"Expected Answer: {result['expected_answer']}")
    print(f"Generated Text: {result['generated_text']}")
    print(f"Text Length (words): {result['length']}")
    print("-" * 50)
