import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def load_model_and_tokenizer(model_name, fine_tuned_model_dir):
    # Load tokenizer and model from the cache
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load fine-tuned model if it exists
    if os.path.isdir(fine_tuned_model_dir):
        print(f"Loading fine-tuned model from {fine_tuned_model_dir}")
        model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_dir)
    else:
        print("Fine-tuned model directory not found. Using base model.")

    # Set pad_token_id to eos_token_id if it is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_tokens):
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    
    # Move input tensors to the same device as the model
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Print input tensors for debugging
    print(f"Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")
    
    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_tokens + input_ids.size(-1),  # Ensure that total length does not exceed token limit
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Print output tensors for debugging
    print(f"Output IDs: {outputs}")

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the generated text
    print("Generated Text:", generated_text)

    # Remove the prompt from the start of the generated text if present
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text

def test_model():
    model_name = "gpt2"
    fine_tuned_model_dir = os.path.expanduser("~/.cache/huggingface/hub/fine-tuned-gpt2")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, fine_tuned_model_dir)

    # Test prompt and max_tokens
    prompt = "why mars is red?"
    max_tokens = 50
    
    # Generate text
    generated_text = generate_text(model, tokenizer, prompt, max_tokens)
    
    # Print the generated text
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    test_model()
