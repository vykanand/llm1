from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Determine the cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Load tokenizer and model from the cache
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)

# Load fine-tuned model if it exists
fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')
if os.path.isdir(fine_tuned_model_dir):
    model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_dir)

# Set pad_token_id to eos_token_id if it is not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def generate_text(prompt, max_tokens):
    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    
    # Move input tensors to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tokens + input_ids.size(-1),  # Ensure that total length does not exceed token limit
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the start of the generated text if present
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.json
        prompt = data.get('text', '')
        max_tokens = int(data.get('max_tokens', 50))  # Default to 50 tokens if not specified
        
        # Check for missing prompt
        if not prompt:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate text based on max_tokens
        generated_text = generate_text(prompt, max_tokens)
        
        return jsonify({'generated_text': generated_text})
    
    except ValueError as ve:
        app.logger.error(f"Value error: {ve}")
        return jsonify({'error': 'Invalid value provided'}), 400
    except Exception as e:
        app.logger.error(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
