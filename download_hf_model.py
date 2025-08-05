from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"  # Replace with actual model if needed
SAVE_DIR = "./pretrained"         # This points to your existing pretrained/ folder

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(SAVE_DIR)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.save_pretrained(SAVE_DIR)

print("✅ Model and tokenizer downloaded into ./pretrained")
