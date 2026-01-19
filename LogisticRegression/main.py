import numpy as np
from src.module import LogisticRegression
from data.preprocessing import preprocess_data

# Prepare Data
data = preprocess_data("data/dataset_noise.csv", target_column="y")
y_raw = data["y"].values
media_y = np.mean(y_raw)
y = (y_raw > media_y).astype(int) # Define 'y' with only 0s and 1s

X = data.drop(columns=["y"]).values
# Configure model with Reegularization L2
model = LogisticRegression(learning_rate=0.001, n_iters=1000, penalty='l2', lambda_param=0.01)

# Train (We use 'y', which is the variable we just defined)
print("Iniciando entrenamiento...")
model.fit(X, y) 

# Evaluation
acc = model.score(X, y)
print(f"Precisión final: {acc * 100:.2f}%")
