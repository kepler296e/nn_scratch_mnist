# Neural Network from Scratch: Digit Recognition
In this project, I have implemented a multi-layer perceptron neural network from scratch in Python to classify handwritten digits from the [MNIST dataset](#mnist-dataset). My goal was to gain a deeper understanding of the underlying concepts by building the model without using deep learning libraries.

#### Screenshots
<img src="screenshots/0.png" width="15%" height="15%"> <img src="screenshots/1.png" width="15%" height="15%"> <img src="screenshots/2.png" width="15%" height="15%"> <img src="screenshots/3.png" width="15%" height="15%">

<img src="screenshots/4.png" width="15%" height="15%"> <img src="screenshots/5.png" width="15%" height="15%"> <img src="screenshots/6.png" width="15%" height="15%"> <img src="screenshots/7.png" width="15%" height="15%">

<img src="screenshots/8.png" width="15%" height="15%"> <img src="screenshots/9.png" width="15%" height="15%"> <img src="screenshots/42.png" width="15%" height="15%">

## Table of Contents
1. [MNIST dataset](#mnist-dataset)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Forward propagation](#forward-propagation)
4. [Loss Function: Cross-entropy](#loss-function-cross-entropy)
5. [Learning and Optimization: Gradient Descent](#learning-and-optimization-gradient-descent)
6. [Real-Time Digit Recognition](#real-time-digit-recognition)
7. [Scratch vs. TensorFlow](#scratch-vs-tensorflow)

#### Cross-entropy loss function
$$J(p,q) = -\sum_{x}p(x)log(q(x))$$
[Why cross-entropy?](#loss-function-cross-entropy)

#### Gradient descent as optimization algorithm
$$w = w - \alpha \frac{\partial L}{\partial w}$$
$$b = b - \alpha \frac{\partial L}{\partial b}$$
[Why gradient descent?](#learning-and-optimization-gradient-descent)

## MNIST dataset
The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) consists of 70,000 images of handwritten digits (0-9). Each image is represented as a 784-vector of pixel values (28x28=784) ranging from 0 to 255.

#### Preprocessing
The dataset is normalized as pixel_value / 255 (max value).

## Neural Network Architecture
#### Input layer
- 784 neurons, one for each pixel.
#### Hidden layers
- Two hidden layers with 25 and 15 neurons respectively.
- Both use [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation.
#### Output layer
- 10 neurons, one for each digit.
- [Softmax](https://en.wikipedia.org/wiki/Softmax_function) activation transforms the nn's output into a probability distribution.

ReLU and Softmax implementation in Python:
```python
def relu(X):
    return np.where(X > 0, X, 0)

def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

#### Hiperparameters
- Epochs: 20.
- Learning Rate: Fixed at 0.01. While more advanced algos like [Adam](https://optimization.cbe.cornell.edu/index.php?title=Adam) could be considered, a learning rate of 0.01 achieves an accuracy of approximately 98%, which is sufficient for practical purposes.
- Weight Initialization: [He initialization](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) for weights, and biases are initialized to 0.

#### After 20 epochs, the loss functions doesn't change much...
<img src="screenshots/dloss.png" width="50%" height="50%">

#### Model implementation in Python:
```python
layers=[784, 25, 15, 10]
model = NN(layers)

print("Parameters:", model.count_params())  # 20175

# train
model.fit(
    X_train,
    y_train,
    epochs=10,
    learning_rate=0.01,
    batch_size=4
)

# evaluate
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)
print("Accuracy:", np.mean(y_pred == y_val))
```
[nn_scratch.py](nn_scratch.py)

## Forward propagation
The `predict(X)` function performs the forward propagation:

```python
def forward_propagation(self, X):
    # input layer
    self.Z[0] = X

    # hidden layers
    for i in range(len(self.layers) - 1):  # 0, 1, 2
        self.Z[i + 1] = relu(np.einsum("bi,bij->bj", self.Z[i], self.W[i]) + self.B[i])  # einsum does @ over the whole batch

    # ouput layer with linear activation
    self.Z[-1] = np.einsum("bi,bij->bj", self.Z[-2], self.W[-1]) + self.B[-1]
    y_pred = softmax(self.Z[-1])

    return y_pred
```

#### Explanation
1. **Input Layer**:
    - Set the input layer as `X`.
2. **Hidden Layers**:
    - For each hidden layer:
        - For each hidden neuron:
            - This neuron relys on all previous layer neurons * its mutual weights: n1w1 + n2w2 + ... + nNwN, and thats literally the definition of dot product: `np.dot(Z[i], W[i].T)`.
            - Add the bias: `+ B[i]`.
            - Apply the activation function: `relu()`.
            - It's done!
3. **Output Layer**:
    - Similar to the hidden layers but with linear activation.
    - Apply the softmax function to get the probabilities of each digit.

## Loss Function: Cross-entropy
Being the generalization of the well-known [log-loss](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression) function for binary classification, cross-entropy is a natural choice for multi-class classification problem.

For each training example:
- `y_pred` is the output of the nn, a 10-vector containing the probabilities of each digit.
- `y_true` is the true label one hot encoded.

It works as follows:
```python
y_pred = self.forward_propagation(X_train[i])
y_true = np.zeros((self.batch_size, self.layers[-1]))
y_true[np.arange(self.batch_size), y_train[i]] = 1  # one hot encode
batch_loss = cross_entropy(y_true, y_pred).mean()  # (batch_size,).mean() => scalar

epoch_loss += batch_loss
```
Where `cross_entropy` is defined as:
```python
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=1)  # 1e-8 to avoid log(0)
```

`np.log(y_pred)` values will be always negative cos `0 < y_pred < 1`.

<img src="screenshots/ln.png" width="50%" height="50%">

so `np.sum()` must be `*-1` to get a positive value.

#### Loss function over epochs
<img src="screenshots/loss.png" width="50%" height="50%">

## Learning and Optimization: Gradient Descent
Neural netorks learns by iteratively adjusting the weights and biases to minimize the loss function. This is done by calculating the partial derivatives of the loss function with respect to the weights and biases, and then updating the weights and biases in the opposite direction of the gradient.

#### Gradient descent
$$w = w - \alpha \frac{\partial L}{\partial w}$$
$$b = b - \alpha \frac{\partial L}{\partial b}$$

Where $\alpha$ is the learning rate.

#### How much does the loss function change when we change the weights and biases?

Using the chain rule:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial w}$$
$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b}$$

Where $z$ is the output of a neuron.

#### Implementation in Python:
```python
# Output layer
dl_dz[-1] = y_pred - y_true
dl_dw[-1] = np.einsum("bi,bj->bji", self.dl_dz[-1], self.Z[-2])
dl_db[-1] = dl_dz[-1]

# Hidden layers
for i in range(-2, -len(layers), -1):
    self.dl_dz[i] = np.einsum("bi,bji->bj", self.dl_dz[i + 1], self.W[i + 1]) * relu_derivative(self.Z[i])
    self.dl_dw[i] = np.einsum("bi,bj->bji", self.dl_dz[i], self.Z[i - 1])
    self.dl_db[i] = self.dl_dz[i]

# Update weights and biases
for i in range(len(self.layers) - 1):  # 0, 1, 2
    self.W[i] -= learning_rate * self.dl_dw[i]
    self.B[i] -= learning_rate * self.dl_db[i]
```

## Real-Time Digit Recognition
After saving the parameters as `scratch_model.npy` using `save_model()`, we can import [nn_scratch.py](nn_scratch.py) and call the `predict()` function over whatever 784-vector we want.

I have built a pretty-basic canvas >.< using [pygame](https://en.wikipedia.org/wiki/Pygame) to make predictions every second.

```python
if frames % FPS == 0 and cells.sum() > 0:
    X = get_X() / 255
    y = model.predict(X)[0]
```
[draw_scratch.py](draw_scratch.py)

#### Controls:
- `Left-click` to draw.
- `Right-click` to erase.
- `R` to reset the canvas.

#### Example:
<img src="screenshots/3.png" width="50%" height="50%">

## Scratch vs. TensorFlow
| | nrows | Epochs | Batch | Accuracy | Time (s) | $\alpha$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Scratch | 5K | 30 | 1 | 0.934 | 28.9791 | 0.01  |
| Scratch NEW | 5K | 10 | 4 | 0.892 | 4.7192 | 0.01 |
| TensorFlow | 5K | 30 | | 0.94 | 6.2679 | Adam(0.001) |

[nn_scratch.py](https://github.com/kepler296e/nn_scratch_mnist/blob/main/nn_scratch.py)

[nn_tf.py](https://github.com/kepler296e/nn_scratch_mnist/blob/main/nn_tf.py)

Over 30 epochs on 5000 training examples and 1000 test examples.