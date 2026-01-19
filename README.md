# Logistic Regression from Scratch

### No Scikit-Learn magic. Just NumPy, Math, and a bit of sweat.

I got tired of typing `from sklearn.linear_model import LogisticRegression` and treating the model like a black box. I wanted to understand **exactly** what happens when a machine "learns."

So, I built a fully functional Logistic Regression library from the ground up. No pre-made `.fit()` methods—just raw Python and linear algebra.

---

## What's Inside?

This project is split into two main parts: **The Engine** (the library) and **The Lab** (the notebook).

### 1. The Engine (`src/`)

This is the core library. It's not a script; it's a reusable module designed to be robust.

* **`src/module.py`**: The heart of the project. Contains the `LogisticRegression` class with manual implementations of:
* **Sigmoid Activation**: Mapping linear inputs to probabilities.
* **Binary Cross-Entropy Loss**: The math that tells the model how wrong it is.
* **Gradient Descent**: The optimization loop that tweaks weights.
* **Regularization (L1 & L2)**: Lasso and Ridge penalties to handle noisy data.


* **`src/preprocessing.py`**: A helper module to clean data and handle normalization (because Gradient Descent *hates* unscaled data).

### 2. The Lab (`Logistic-Regression.ipynb`)

This is where I research, develop and put the engine to the test.

* I benchmark the model against the **Breast Cancer Wisconsin** dataset.
* I implement a custom **Grid Search** and **K-Fold Cross-Validation** (again, from scratch) to find the best hyperparameters.
* It includes visualizations of the Loss Curve to prove the model is actually converging.

---

## Key Features

This isn't just a toy implementation. I spent time refining it to handle real-world edge cases:

* **Numerical Stability**: I implemented clipping mechanisms in the sigmoid and loss functions. If you've ever seen `RuntimeWarning: overflow encountered in exp`, you know the pain. My implementation handles this gracefully.
* **Regularization**: Supports both **L1** (for feature selection) and **L2** (for weight control) penalties.
* **Vectorization**: The code relies heavily on `numpy` matrix operations (`np.dot`), making it significantly faster than using Python `for` loops.

---

## How to Run It

### Prerequisites

You just need the basics. No heavy ML frameworks required for the logic.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Note: scikit-learn is ONLY used for loading datasets and splitting data, not for the model itself!

```

### Usage

You can use my class exactly like you would use Scikit-Learn:

```python
from src.module import LogisticRegression

# Initialize
# We can use L2 regularization to prevent overfitting on noisy data
model = LogisticRegression(learning_rate=0.01, n_iters=1000, penalty='l2', lambda_param=0.1)

# Train
# Watch the gradient descent work its magic
model.fit(X_train, y_train)

# Predict & Score
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

```

---

## The Math Behind the Code

The core of this library relies on minimizing the **Log Loss** function:

$$J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} [ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) ]$$

But the real challenge was the **Backpropagation**. Calculating the derivatives manually:

$$\frac{\partial J}{\partial w}=\frac{1}{m} X^T (\hat{y} - y)+\frac{\lambda}{m} \mathrm{sgn}(w)$$

*(That last term is the L1 regularization derivative I added to handle feature selection).*

---

**Feel free to fork this, break it, and fix it. That's the only way to learn.**
