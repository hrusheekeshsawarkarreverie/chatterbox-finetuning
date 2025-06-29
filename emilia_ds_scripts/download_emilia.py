from datasets import load_dataset


data_files = {
    "de": "Emilia-YODAS/DE/*.tar",
    "fr": "Emilia-YODAS/FR/*.tar",
    "ja": "Emilia-YODAS/JA/*.tar"
}
CHATTERBOX_PROJECT = "chatterbox-project"

dataset = load_dataset(
        "amphion/Emilia-Dataset",
        data_files=data_files,
        split=None,
        # streaming=True,
        cache_dir=CHATTERBOX_PROJECT,
    )