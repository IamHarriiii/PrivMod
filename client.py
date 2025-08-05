from flwr.client import NumPyClient, start_client
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding

# Import your model and tokenizer
from models.text_model import TextModerationModel
from models.tokenizer_utils import get_tokenizer

# Load and prepare dataset
def load_data(tokenizer, num_samples=100):
    dataset = load_dataset("imdb").shuffle(seed=42)
    train_data = dataset["train"].select(range(num_samples))
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = train_data.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_datasets

class ModerationClient(NumPyClient):
    def __init__(self):
        self.model = TextModerationModel().get_model()
        self.tokenizer = get_tokenizer()
        self.train_data = load_data(self.tokenizer)

    def get_parameters(self, config):
        """Return model parameters as list of NumPy arrays"""
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train locally and return updated parameters"""
        self.set_parameters(parameters)
        
        # Create DataLoader
        dataloader = DataLoader(self.train_data, batch_size=8)

        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        # Training loop (simplified)
        self.model.train()
        for batch in dataloader:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label'].float()
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Training complete.")
        return self.get_parameters(config), len(dataloader), {}

    def evaluate(self, parameters, config):
        """Evaluate locally and return metrics"""
        self.set_parameters(parameters)
        # Optional: Implement evaluation
        return float(0.5), 10, {"accuracy": 0.9}

if __name__ == "__main__":
    # Start Flower client
    start_client(
        server_address="server:8080",
        client=ModerationClient()
    )