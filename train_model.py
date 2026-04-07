from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'flower_model.joblib')

print("Model saved successfully!")