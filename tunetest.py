import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
import language_tool_python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the cache directory and model paths
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_name = "gpt2"  # Using base GPT-2 model for testing
fine_tuned_model_dir = os.path.join(cache_dir, 'fine-tuned-gpt2')

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)  # Changed to base model for testing

# Set pad_token_id if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Initialize language tool for grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

def generate_text(prompt, max_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_tokens + len(inputs['input_ids'][0]),
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.85,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debug print statements
    print("Prompt:")
    print(prompt)
    print("\nGenerated Text:")
    print(generated_text)
    print("-" * 50)

    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def compute_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def check_grammar(text):
    matches = language_tool.check(text)
    return len(matches)  # Number of grammar issues

def evaluate_performance(data_file, max_tokens=50):
    with open(data_file, 'r') as file:
        data = json.load(file)
    
    similarities = []
    grammar_errors = []
    lengths = []
    deviations = []

    for item in data:
        prompt = item['question']
        expected_answer = item['answer']
        generated_text = generate_text(prompt, max_tokens)
        text_length = len(generated_text.split())
        
        similarity = compute_similarity(expected_answer, generated_text)
        grammar_error_count = check_grammar(generated_text)
        deviation = abs(len(expected_answer.split()) - text_length)
        
        similarities.append(similarity)
        grammar_errors.append(grammar_error_count)
        lengths.append(text_length)
        deviations.append(deviation)

    # Plot the metrics
    plt.figure(figsize=(15, 10))

    # Similarity Plot
    plt.subplot(2, 2, 1)
    plt.hist(similarities, bins=20, color='green', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores')

    # Grammar Errors Plot
    plt.subplot(2, 2, 2)
    plt.hist(grammar_errors, bins=20, color='red', edgecolor='black')
    plt.xlabel('Number of Grammar Errors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Grammar Errors')

    # Text Length Plot
    plt.subplot(2, 2, 3)
    plt.hist(lengths, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Length of Generated Text (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Generated Text Lengths')

    # Deviation Plot
    plt.subplot(2, 2, 4)
    plt.hist(deviations, bins=20, color='purple', edgecolor='black')
    plt.xlabel('Deviation from Expected Answer Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Answer Deviation')

    plt.tight_layout()
    plt.savefig('performance_metrics.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

    # Explanations for the plots
    print("\nPlot Explanations:")
    print("1. **Distribution of Similarity Scores**: This plot shows how similar the generated answers are to the expected answers based on cosine similarity scores. Higher scores indicate better similarity. If the scores are generally low, the model's responses might not be closely matching the expected answers.")
    print("2. **Distribution of Grammar Errors**: This histogram illustrates the number of grammar errors detected in the generated texts. Fewer errors suggest better grammatical quality. A high number of errors may indicate issues with the model's ability to generate grammatically correct text.")
    print("3. **Distribution of Generated Text Lengths**: This plot represents the lengths of the generated texts in terms of word count. It helps to understand the verbosity of the generated answers. If the lengths vary significantly, it could mean the model is generating excessively short or long responses.")
    print("4. **Distribution of Answer Deviation**: This shows how much the length of the generated text deviates from the expected answer length. Smaller deviations indicate more precise text generation. Larger deviations might suggest that the model is not generating responses with appropriate length.")

    # Overall analysis
    avg_similarity = np.mean(similarities)
    avg_grammar_errors = np.mean(grammar_errors)
    avg_length = np.mean(lengths)
    avg_deviation = np.mean(deviations)

    print("\nOverall Performance Analysis:")
    print(f"Average Similarity Score: {avg_similarity:.2f}")
    print(f"Average Number of Grammar Errors: {avg_grammar_errors:.2f}")
    print(f"Average Length of Generated Text: {avg_length:.2f} words")
    print(f"Average Deviation from Expected Answer Length: {avg_deviation:.2f} words")

    if avg_similarity < 0.5:
        print("The model's generated responses are generally not similar to the expected answers. Improvement in the model's training or fine-tuning might be required.")
    else:
        print("The model's generated responses are fairly similar to the expected answers.")

    if avg_grammar_errors > 5:
        print("The generated texts have a high number of grammar errors. Enhancing the model's ability to generate grammatically correct sentences could be beneficial.")
    else:
        print("The generated texts have a relatively low number of grammar errors.")

    if avg_deviation > 10:
        print("The model's responses have a significant deviation in length compared to expected answers. Adjusting the model's parameters or prompt length may help in generating more appropriately sized responses.")
    else:
        print("The model's responses are of appropriate length compared to expected answers.")

    results = []
    for item in data:
        prompt = item['question']
        expected_answer = item['answer']
        generated_text = generate_text(prompt, max_tokens)
        results.append({
            'prompt': prompt,
            'expected_answer': expected_answer,
            'generated_text': generated_text
        })
    
    return results

# Path to the JSON file containing questions and answers
data_file = '/kaggle/input/t-small/t-small.json'

# Evaluate the performance
performance_results = evaluate_performance(data_file)

# Print the results
for result in performance_results:
    print("\n" + "="*50)
    print(f"Prompt:")
    print(result['prompt'])
    print(f"\nExpected Answer:")
    print(result['expected_answer'])
    print(f"\nGenerated Text:")
    print(result['generated_text'])
    print("="*50)
