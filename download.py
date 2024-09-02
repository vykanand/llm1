import os
from huggingface_hub import HfApi, hf_hub_download

def save_token(token, file_path):
    """Save the API token to a file."""
    with open(file_path, 'w') as f:
        f.write(token)
    print("API token saved successfully.")

def load_token(file_path):
    """Load the API token from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    return None

def main():
    # Define the path to store the API token
    token_file_path = os.path.expanduser("~/.huggingface_token")
    
    # Define the default directory to save the model
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Load the token if it exists
    api_key = load_token(token_file_path)

    if api_key:
        print("API token loaded from file.")
    else:
        # Prompt user for Hugging Face API key and save it
        api_key = input("Enter your Hugging Face API key: ").strip()
        save_token(api_key, token_file_path)

    # Prompt user for repository details
    repo_id = input("Enter the repository ID (e.g., 'vykanand/finegpt1'): ").strip()

    # Validate the cache directory path
    if not os.path.isdir(cache_dir):
        print(f"Directory '{cache_dir}' does not exist. Creating it now.")
        os.makedirs(cache_dir, exist_ok=True)

    # Initialize the Hugging Face API
    api = HfApi()

    try:
        # List files in the repository
        files = api.list_repo_files(repo_id=repo_id, token=api_key)
        
        # Download each file
        for file in files:
            print(f"Downloading {file}...")
            hf_hub_download(repo_id=repo_id, filename=file, token=api_key, cache_dir=cache_dir)
        
        print(f"Model successfully downloaded to {cache_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
