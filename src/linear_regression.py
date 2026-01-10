import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearRegressionScratch:
    """
    Linear Regression with Gradient Descent

    Model: y = X @ w + b
    Cost: MSE = (1/m) * sum((y_pred - y)^2)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape

        # Initialize parameters
        self.weights = np.zeros(n)
        self.bias = 0

        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute cost
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Gradients
            dw = (1 / m) * (X.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Clip gradients to prevent explosion



            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")

        return self

    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias

    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PolynomialRegressionScratch:
    """
    Polynomial Regression using Normal Equation (closed-form solution)
    No gradient descent issues with high dimensions
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Transform to polynomial and solve with normal equation"""
        X_poly = self.poly_features.fit_transform(X)
        m, n = X_poly.shape

        # Add bias column
        X_b = np.c_[np.ones((m, 1)), X_poly]

        # Normal equation: θ = (X^T X)^-1 X^T y
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

        print(f"✓ Fitted with {n} polynomial features")
        return self

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return X_poly @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class PolynomialRegressionGradientDescent:
    """
    Polynomial Regression with Gradient Descent
    Uses optimized learning rate for high-dimensional features
    """

    def __init__(self, degree=2, learning_rate=0.06, n_iterations=2000):
        self.degree = degree
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        """Transform to polynomial and train with gradient descent"""
        X_poly = self.poly_features.fit_transform(X)
        m, n = X_poly.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = X_poly @ self.weights + self.bias
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            dw = (1 / m) * (X_poly.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if (i + 1) % 200 == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")

        print(f"✓ Trained with {n} polynomial features")
        return self

    def predict(self, X):
        X_poly = self.poly_features.transform(X)
        return X_poly @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

