import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# helper function
def sigmoid(x):
    x[x >= 50] = 50
    x[x <= -50] = -50
    return 1.0 / (1 + np.exp(-x))

from matplotlib import pyplot as plt

def plot_stats(stats):
    plt.subplot(3, 1, 1)
    plt.plot(stats['grad_magnitude_history'])
    plt.title('Gradient magnitude history (W1)')
    plt.xlabel('Iteration')
    plt.ylabel('||W1||')
    plt.ylim(0, np.minimum(100,np.max(stats['grad_magnitude_history'])))
    plt.subplot(3, 1, 2)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.ylim(0, 100)
    plt.subplot(3, 1, 3)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

"""
    TwoLayerMLP is Code that's taken from class
    we do not own this code
"""

class TwoLayerMLP(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4, activation='relu'):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    self.activation = activation

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
#    print("hello")
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    _, C = W2.shape
    N, D = X.shape

#    print(X.shape)
#    print(W1.shape)
    # Compute the forward pass
    scores = None
    #############################################################################
    # print(b1.shape)
    # print("HELLO")
    z1 = np.dot(X, W1)
    z1 = z1 + b1  # 1st layer activation, N*H


    # 1st layer nonlinearity, N*H
    if self.activation is 'relu':
        hidden = np.maximum(0, z1)        
        
    elif self.activation is 'sigmoid':
        hidden = sigmoid(z1)
    
    elif self.activation is 'tanh':
        hidden = np.tanh(z1)
    
    elif self.activation is 'leaky_relu':
        alpha = 0.01
        hidden = np.maximum(z1, alpha*z1)
    else:
        raise ValueError('Unknown activation type')
        
    scores = np.dot(hidden, W2) + b2  # 2nd layer activation, N*C
#    if self.activation is 'tanh':
#        scores = np.exp(scores) # tanh returns scores for [-1,1]
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # cross-entropy loss with log-sum-exp
    A = np.max(scores, axis=1) # N*1
    F = np.exp(scores - A.reshape(N, 1))  # N*C
    P = F / np.sum(F, axis=1).reshape(N, 1)  # N*C
    # print(type(P[0][0]))
    y = y.astype('int')
    # print(y.shape)
    # print(scores)
    # print(type(scores[0][0]))
    # print((-np.choose(y, scores.T)).shape)
    loss = np.mean(-np.choose(y, scores.T) + np.log(np.sum(F, axis=1)) + A)
    # add regularization terms
    loss += 0.5 * reg * np.sum(W1 * W1)
    loss += 0.5 * reg * np.sum(W2 * W2)

    # Backward pass: compute gradients
    grads = {}

    #############################################################################

    # output layer
    dscore = P - (np.tile(np.arange(C), (N, 1)) == y.reshape(N, 1))  # N*C
    dW2 = np.dot(hidden.T, dscore)/N + reg * W2  # H*C
    db2 = np.mean(dscore, axis=0)  # C

    # hidden layer
    dhidden = np.dot(dscore, W2.T)  # N*H
    if self.activation is 'relu':
        dz1 = dhidden
        dz1[z1 <= 0] = 0

    elif self.activation is 'sigmoid':
        dz1 = (hidden*(1-hidden)) * dhidden

    elif self.activation is 'tanh':
        dz1 = dhidden*(1-np.power(hidden, 2))
    
    elif self.activation is 'leaky_relu':
        dz1 = dhidden
        dz1[z1 < 0] = 0.01
    else:
        raise ValueError('Unknown activation type')

    dW1 = np.dot(X.T, dz1)/N + reg * W1  # D*H
    db1 = np.mean(dz1, axis=0)  # D
    #############################################################################

    grads['W2'] = dW2 + reg*W2
    grads['b2'] = db2
    grads['W1'] = dW1 + reg*W1
    grads['b1'] = db1
    return loss, grads


  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_epochs=10,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(int(num_train/batch_size), 1)
    epoch_num = 0

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    grad_magnitude_history = []
    train_acc_history = []
    val_acc_history = []

    np.random.seed(1)
    for epoch in range(num_epochs):
        # fixed permutation (within this epoch) of training data
        perm = np.random.permutation(num_train)

        # go through minibatches
        for it in range(iterations_per_epoch):
            X_batch = None
            y_batch = None

            idx = perm[it*batch_size:(it+1)*batch_size]
            X_batch = X[idx, :]
            y_batch = y[idx]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            for param in self.params:
                self.params[param] -= grads[param] * learning_rate

            # record gradient magnitude (Frobenius) for W1
            grad_magnitude_history.append(np.linalg.norm(grads['W1']))

        # Every epoch, check train and val accuracy and decay learning rate.
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        if verbose:
            print('Epoch %d: loss %f, train_acc %f, val_acc %f'%(
                epoch+1, loss, train_acc, val_acc))

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'grad_magnitude_history': grad_magnitude_history, 
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################

    return y_pred

def network(X, y, X_validation, activation):
    print("Currently running on ", activation)
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    input_size = X_train.shape[1]
    hidden_size = 10  # number of neurons in each hidden layer
    num_classes = 2  # number of classes C; in our case 0 and 1

    # This cell calls the neural network
    net = TwoLayerMLP(input_size, hidden_size, num_classes, activation=activation, std=1e-1)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                                      num_epochs=20, batch_size=100,
                                      learning_rate=1e-3,  learning_rate_decay=0.95,
                                      reg=0.5, verbose=True)
    # Predict on the training set
    tra_1 = net.predict(X_train)
    train_acc = (tra_1 == y_train).mean()
    print(activation+'final training accuracy: ', train_acc)
    print("Any 1's in train: ", sum(tra_1))

    # Predict on the validation set
    val_1 = net.predict(X_val)
    val_acc = (val_1 == y_val).mean()
    print(activation+' final validation accuracy: ', val_acc)
    print("Any 1's in validation: ", sum(val_1))

    # Predict on the test set
    test_acc = (net.predict(X_test) == y_test).mean()
    print(activation+' test accuracy: ', test_acc)

    # show stats and visualizations
    plot_stats(stats)

    pred = net.predict(X_validation)
    print('Number of no show out of 222 patients', sum(pred))


def main():
    X = np.load('xTrain.npy')
    y = np.load('yTrain.npy')
    X_val = np.load('xVal.npy')
    network(X, y, X_val, 'sigmoid')
    network(X, y, X_val, 'relu')
    network(X, y, X_val, 'tanh')
    network(X, y, X_val, 'leaky_relu')



if __name__ == '__main__':
    main()
