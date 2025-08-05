from transformers import AutoModelForSequenceClassification

class TextModerationModel:
    def __init__(self, model_name_or_path="./models/pretrained"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=True)

    def get_model(self):
        return self.model