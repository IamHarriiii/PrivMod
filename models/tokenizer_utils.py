from transformers import AutoTokenizer

def get_tokenizer(model_name_or_path="./models/pretrained"):
    return AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)