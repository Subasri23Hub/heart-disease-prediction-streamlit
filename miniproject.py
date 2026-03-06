import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix

# load dataset
df = pd.read_csv("heart.csv")

# encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# hyperparameters
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs"]
}

# gridsearch
grid = GridSearchCV(pipeline, param_grid, cv=3)

# train
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# predictions
y_pred = best_model.predict(X_test)

# metrics
print("Best parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# plot confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()