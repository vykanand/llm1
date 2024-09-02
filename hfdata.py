import os
import json
import pandas as pd
from datasets import load_dataset
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def display_sample_data(df, num_samples=10):
    """Display the first few rows of the DataFrame."""
    print("\nSample Data:")
    print(df.head(num_samples))
    print()

def clean_and_normalize_text(text):
    """Clean special characters and normalize the text."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters except spaces
    text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
    text = text.strip()                  # Remove leading and trailing spaces
    text = text.lower()                 # Convert to lowercase
    
    return text

def convert_to_custom_format(df, question_field, answer_field):
    """Convert the DataFrame to the desired format."""
    converted_data = []
    
    for _, row in df.iterrows():
        question = row.get(question_field, '')
        answer = row.get(answer_field, '')
        
        # Clean and normalize the text
        question = clean_and_normalize_text(question)
        answer = clean_and_normalize_text(answer)
        
        # Add to the converted data list in the required format
        converted_data.append({
            "question": question,
            "answer": answer
        })
    
    return converted_data

def validate_json(data):
    """Check if the data can be serialized to JSON."""
    try:
        json.dumps(data)  # Attempt to serialize to JSON to check validity
        return True
    except (TypeError, OverflowError):
        return False

def ensure_data_downloaded(dataset_name, config_name, split, data_dir='data'):
    """Ensure that dataset files are downloaded and available."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Construct file paths
    csv_file_path = os.path.join(data_dir, f"{dataset_name.replace('/', '-')}_{config_name}_{split}.csv")
    parquet_file_path = os.path.join(data_dir, f"{dataset_name.replace('/', '-')}_{config_name}_{split}.parquet")

    # Check if either file exists
    if os.path.exists(csv_file_path):
        print(f"CSV dataset file already exists: {csv_file_path}")
        return csv_file_path, 'csv'
    elif os.path.exists(parquet_file_path):
        print(f"Parquet dataset file already exists: {parquet_file_path}")
        return parquet_file_path, 'parquet'
    else:
        print(f"Downloading dataset: {dataset_name} with config: {config_name} and split: {split}...")
        dataset = load_dataset(dataset_name, config_name, split=split)
        df = dataset.to_pandas()
        
        # Save datasets in both formats
        df.to_csv(csv_file_path, index=False)
        df.to_parquet(parquet_file_path, index=False)
        print(f"Dataset downloaded and saved to {csv_file_path} and {parquet_file_path}.")
        return parquet_file_path, 'parquet'  # Default to parquet format if both are saved

def compute_text_similarity(text1, text2):
    """Compute the cosine similarity between two texts."""
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def calculate_similarity_scores(data):
    """Calculate similarity scores between expected and generated answers."""
    data['similarity'] = data.apply(lambda row: compute_text_similarity(row['expected_answer'], row['generated_answer']), axis=1)
    return data

def plot_similarity_scores(similarity_scores):
    """Plot the distribution of text similarity scores."""
    plt.figure(figsize=(10, 5))
    plt.hist(similarity_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Similarity Scores')
    plt.grid(True)
    plt.savefig('similarity_scores.png')  # Save the plot as a PNG file
    plt.show()  # Display the plot

def main(dataset_name, config_name, split='train', data_dir='data'):
    try:
        # Ensure data is downloaded and detect format
        dataset_file_path, file_format = ensure_data_downloaded(dataset_name, config_name, split, data_dir)
        
        # Load the dataset
        if file_format == 'csv':
            df = pd.read_csv(dataset_file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(dataset_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Display the first 10 entries
        display_sample_data(df, num_samples=10)
        
        # Ask user to specify question and answer fields
        question_field = input("Enter the column name for 'question': ").strip()
        answer_field = input("Enter the column name for 'answer': ").strip()
        
        # Convert the dataset to the custom format
        converted_data = convert_to_custom_format(df, question_field, answer_field)
        
        # Convert to DataFrame and calculate similarity scores
        data_df = pd.DataFrame(converted_data)
        # If you need to calculate similarity scores, adjust the function call accordingly
        # data_df = calculate_similarity_scores(data_df)
        
        # Confirm before writing to file
        display_sample_data(data_df, num_samples=10)  # Display converted data
        confirmation = input("Do you want to save the converted data to 'tuning.json'? (yes/no): ").strip().lower()
        
        if confirmation == 'yes':
            # Validate JSON data before saving
            if validate_json(converted_data):
                # Save the converted data to a local JSON file
                output_file = 'tuning.json'
                with open(output_file, 'w') as f:
                    json.dump(converted_data, f, indent=4)
                
                print(f"Data has been successfully converted and saved to {output_file}.")
            else:
                print("Converted data is not valid JSON. Aborting save.")
        
            # Plot similarity scores (optional, based on your needs)
            # plot_similarity_scores(data_df['similarity'])
        else:
            print("Data conversion was aborted.")
    
    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == '__main__':
    # Replace 'your_dataset' with the actual dataset name
    # Replace 'config_name' with the actual configuration name if required
    main('Omkar7/Medical_data', 'default', split='train')
