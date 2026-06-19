from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model saved!")