from huggingface_hub import snapshot_download
model_id="mardonbekhazratov/gpt2-trained-from-scratch"
snapshot_download(repo_id=model_id, local_dir="model",
                  local_dir_use_symlinks=False, revision="main")
