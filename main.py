import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset (file must be inside "New folder (2)")
df = pd.read_csv("sonar.csv", header=None)

# Convert labels: R -> 0, M -> 1
df[60] = df[60].map({'R': 0, 'M': 1})

# Split features and labels
X = df.drop(60, axis=1)
y = df[60]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test with one sample
sample = X.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

if prediction[0] == 0:
    print("Prediction: Rock")
else:
    print("Prediction: Mine")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC()
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(name, accuracy_score(y_test, m.predict(X_test)))
    from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
model = SVC(kernel='rbf', C=10, gamma='scale')
SVC(class_weight='balanced')
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

model = SVC()

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation accuracy:", scores.mean())