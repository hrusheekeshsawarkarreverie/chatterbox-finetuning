from huggingface_hub import snapshot_download

repo_name = "ResembleAI/chatterbox"

snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir="chatterbox_weights")

