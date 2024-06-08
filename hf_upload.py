from huggingface_hub import HfApi
from dotenv import load_dotenv
import os

load_dotenv()

hf_api = os.getenv("HF_API_KEY")
if hf_api is None:
    raise ValueError("HF_API_KEY is not set in the environment variables")

api = HfApi()

api.upload_folder(
    folder_path="./data/faces",
    path_in_repo=".",
    repo_id="8clabs/sdxl-faces",
    repo_type="dataset",
    token=hf_api,
)