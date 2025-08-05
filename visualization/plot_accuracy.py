import matplotlib.pyplot as plt

rounds = [1, 2, 3, 4, 5]
acc = [0.75, 0.80, 0.83, 0.85, 0.87]

plt.plot(rounds, acc, label="Accuracy")
plt.xlabel("FL Rounds")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Federated Learning Rounds")
plt.legend()
plt.savefig("accuracy_plot.png")