import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv("data/iris.csv")

# 2. Features (X) and target (y)
X = data.drop("species", axis=1)
y = data["species"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Load the saved model
model = joblib.load("model/iris_model.pkl")

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f" Model accuracy: {accuracy:.2%}")
