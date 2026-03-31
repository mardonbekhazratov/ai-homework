from huggingface_hub import HfApi
api = HfApi()

model_id = "mardonbekhazratov/gpt2-trained-from-scratch-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=r"gpt2-from-scratch.gguf",
    path_in_repo=r"gpt2-from-scratch.gguf",
    repo_id=model_id,
)