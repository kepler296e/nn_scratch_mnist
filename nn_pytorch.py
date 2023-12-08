import torch
import torch.nn as nn
import nn_data
import time


def main():
    # Load data
    X_train, y_train, X_val, y_val = nn_data.load_data(pytorch=True)

    # Build model
    model = NN().to(nn_data.DEVICE)
    print(model.count_params(), "parameters")

    # Train
    start_train_time = time.time()
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        eval_every=1,
        lr=0.01,
        batch_size=128,
    )
    print("Train time", time.time() - start_train_time)

    torch.save(model.state_dict(), "models/pytorch.pth")


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
        )
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X):
        logits = self.linear_relu_stack(X)
        return logits

    def fit(self, X_train, y_train, X_val, y_val, epochs, eval_every, lr, batch_size):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss = 0
            for i in range(0, len(X_train) - len(X_train) % batch_size, batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Forward pass
                logits = self(X_batch)
                loss = loss_fn(logits, y_batch)
                train_loss += loss.item() / (len(X_train) / batch_size)

                # Backward pass
                optimizer.zero_grad()  # this resets the gradients of all model parameters
                loss.backward()  # computes the gradient of the loss with respect to model parameters
                optimizer.step()  # updates the parameters based on the current gradient

            if (epoch + 1) % eval_every == 0:
                val_loss = loss_fn(self(X_val), y_val).item()  # item() returns the value of this tensor as a standard Python number
                print(f"Epoch {epoch + 1}/{epochs}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")


if __name__ == "__main__":
    main()
