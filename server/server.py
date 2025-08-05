from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2
)

if __name__ == "__main__":
    config = ServerConfig(num_rounds=5)
    start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy
    )