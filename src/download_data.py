import os
import requests
import zipfile
import io

def download_data(output_dir='data/raw'):
    """
    Downloads the Credit Card Default dataset from UCI.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    print(f"Downloading from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(output_dir)
            print(f"Extracted to {output_dir}")
            
        # Check files
        print("Files in data/raw:")
        for f in os.listdir(output_dir):
            print(f" - {f}")

    except Exception as e:
        print(f"Error downloading or extracting data: {e}")

if __name__ == "__main__":
    download_data()
