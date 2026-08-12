from huggingface_hub import snapshot_download, login
from huggingface_hub.utils import RepositoryNotFoundError

# 1. Authenticate (This will prompt for a token if you aren't logged in)
try:
    # Check if a token is already saved; if not, prompt the user
    login()
except Exception as e:
    print(f"Login failed: {e}")

# 2. Specify the repository ID and the local directory path
repo_id = "aggr8/COL775-A2-Clevr-Extended-100k"
#local_dir = "./my_local_dataset_folder"
local_dir = "/mnt/bigdisk/Others/775A2/my_local_dataset_folder"

print(f"Starting download from {repo_id}...")

# 3. Download the files
try:
    snapshot_download(
        repo_id=repo_id, 
        local_dir=local_dir, 
        repo_type="dataset"
    )
    print(f"Download complete! Files saved to: {local_dir}")
except RepositoryNotFoundError:
    print(f"Error: Repository '{repo_id}' not found. Make sure you have access.")
except Exception as e:
    print(f"An error occurred: {e}")