## brahmaanu_llm/huggingface/hfhub_upload.py

## I just want to show the code I used to upload the checkpoints I got after SFT, GGUF etc (the full folder) to HF Hub


import os
from huggingface_hub import HfApi, create_repo, upload_folder

# put your write token here once, or set it as an env var before running the notebook
os.environ["HF_TOKEN"] = ... # replace with oyur token string
api = HfApi(token=os.environ["HF_TOKEN"])

# choose a name; keep it private if you like
username = api.whoami()["name"]
repo_id  = f"{username}/bro-sft" ## My HF repo

#create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=os.environ["HF_TOKEN"])

# uploads the full folder: export/{qlora-1gpu, lora-zero3-2gpu, ...}
upload_folder(
    repo_id=repo_id,    
    repo_type="model", ## Can be either model or dataset etc
    folder_path=..., ## Replace with your local folder containing everything
    commit_message="Upload of observatory docs",
    token=os.environ["HF_TOKEN"]
)

# quick sanity check
api.list_repo_files(repo_id, repo_type="model")
