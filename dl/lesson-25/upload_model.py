from huggingface_hub import HfApi
api = HfApi()

model_id = "mardonbekhazratov/bert-base-uncased-fine-tuned-gguf"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj=r"bert-base-uncased-fine-tuned.gguf",
    path_in_repo=r"bert-base-uncased-fine-tuned.gguf",
    repo_id=model_id,
)