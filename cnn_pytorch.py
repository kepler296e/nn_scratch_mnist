import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nn_data
import time


def main():
    # Load data
    X_train, y_train, X_val, y_val = nn_data.load_data(pytorch=True)

    # Transform flat data to 2D images
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)

    # Build model
    model = CNN().to(nn_data.DEVICE)
    print(model.count_params(), "parameters")

    # Train
    start_train_time = time.time()
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=1,
        eval_every=1,
        lr=0.01,
        batch_size=128,
    )
    print("Train time", time.time() - start_train_time)

    torch.save(model.state_dict(), "models/cnn_torch.pth")

    # Test accuracy
    accuracy = calculate_accuracy(model, X_val, y_val)
    print("Accuracy:", accuracy)


def calculate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        _, predicted = torch.max(logits, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / len(y)
    model.train()
    return accuracy


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=2)
        X = X.reshape(-1, 32 * 7 * 7)
        X = self.fc1(X)
        return X

    def fit(self, X_train, y_train, X_val, y_val, epochs, eval_every, lr, batch_size):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % eval_every == 0:
                val_loss = loss_fn(self(X_val), y_val).item()
                print(f"Epoch {epoch + 1}/{epochs}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

    def predict(self, X):
        with torch.no_grad():
            logits = self(X)
            y_pred = F.softmax(logits, dim=1)
            return y_pred.cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(path):
    model = CNN()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    main()
