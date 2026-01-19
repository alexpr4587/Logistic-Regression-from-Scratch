import numpy as np
import itertools

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, penalty=None, lambda_param=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, z):
        # Clip to avoid math errors (Overflow)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        # Binary Cross-Entropy Loss
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # We save the loss to see the progress
            self.losses.append(self._compute_loss(y, y_predicted))

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Regularization L1 or L2
            if self.penalty == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            elif self.penalty == 'l2':
                dw += (self.lambda_param / n_samples) * 2 * self.weights

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y).reshape(-1)
        return np.mean(y_pred == y_true)

# Utility functions for validation

def k_fold_cross_validation(model_class, X, y, k=5, **model_params):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_size = n_samples // k
    accuracies = []

    for i in range(k):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        model = model_class(**model_params)
        model.fit(X[train_idx], y[train_idx])
        accuracies.append(model.score(X[val_idx], y[val_idx]))
    return np.mean(accuracies)

def grid_search(model_class, X, y, param_grid, cv=5):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_score, best_params = -1, None

    for params in combinations:
        score = k_fold_cross_validation(model_class, X, y, k=cv, **params)
        if score > best_score:
            best_score, best_params = score, params
    return best_params, best_score
