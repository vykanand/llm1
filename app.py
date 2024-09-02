from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define cache directory
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

def list_models_in_cache(cache_dir):
    """
    List all models available in the local Hugging Face cache directory.
    
    Args:
        cache_dir (str): The directory to scan for models.
        
    Returns:
        model_dirs (list): A list of model directories.
    """
    model_dirs = []
    
    for root, dirs, files in os.walk(cache_dir):
        # Check if the directory contains model files
        if any(file.endswith(('pytorch_model.bin', 'tf_model.h5', 'model.safetensors')) for file in files):
            model_dirs.append(root)
    
    return model_dirs

def load_model_and_tokenizer(model_name_or_path):
    """
    Load model and tokenizer dynamically from Hugging Face.
    
    Args:
        model_name_or_path (str): The name or path of the model.
        
    Returns:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        model (PreTrainedModel): The model.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    # Set pad_token_id to eos_token_id if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer, model

def get_device():
    """
    Get the appropriate device for model loading (GPU or CPU).
    
    Returns:
        device (torch.device): The device to use.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# List all models in cache and display them
model_dirs = list_models_in_cache(cache_dir)

if not model_dirs:
    logger.error("No models found in cache. Exiting.")
    exit()

logger.info("Available models:")
for i, model_dir in enumerate(model_dirs):
    logger.info(f"{i}: {model_dir}")

# Ask user to select a model
try:
    selected_index = int(input("Select a model by index: "))
    if selected_index < 0 or selected_index >= len(model_dirs):
        raise ValueError("Invalid index selected.")
    model_name = model_dirs[selected_index]
except (ValueError, IndexError) as e:
    logger.error(f"Error: {e}. Exiting.")
    exit()

# Load selected model and tokenizer
tokenizer, model = load_model_and_tokenizer(model_name)

# Get the device (GPU or CPU)
device = get_device()
model.to(device)

# Set up mixed precision
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# Set up asynchronous executor
executor = ThreadPoolExecutor(max_workers=4)  # Adjust max_workers based on your system

async def generate_text_async(prompt, max_tokens):
    """
    Generate text asynchronously.
    
    Args:
        prompt (str): The input text to generate from.
        max_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        generated_text (str): The generated text.
    """
    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    
    # Move input tensors to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Generate text
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_tokens + input_ids.size(-1),  # Ensure that total length does not exceed token limit
                num_return_sequences=1,
                temperature=0.7,  # Lower temperature for less randomness
                top_p=0.9,        # Increase top_p to include more diverse tokens
                no_repeat_ngram_size=2,  # Prevent repetition of n-grams
                early_stopping=True,     # Stop early if the model is confident
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
    """
    Handle POST request to generate text based on a prompt.
    
    Returns:
        JSON response with generated text.
    """
    try:
        # Get JSON data from the POST request
        data = request.json
        prompt = data.get('text', '')
        max_tokens = int(data.get('max_tokens', 50))  # Default to 50 tokens if not specified
        
        # Check for missing prompt
        if not prompt:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate text asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        generated_text = loop.run_until_complete(generate_text_async(prompt, max_tokens))
        
        return jsonify({
            'generated_text': generated_text,
            'prompt': prompt,
            'max_tokens': max_tokens
        })
    
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        return jsonify({'error': 'Invalid value provided'}), 400
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
