import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_excel("mydata_dame.xlsx")
X = data[['Voltage (mV)']].values  # Feature
y = data['Salt taste type'].values   # Target

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

# Scale the voltage data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_scaled, y_train)

# Validation Accuracy
val_accuracy = knn.score(X_val_scaled, y_val)
print("Validation accuracy:", val_accuracy)

# Test Accuracy
test_accuracy = knn.score(X_test_scaled, y_test)
print("Test accuracy:", test_accuracy)

# Save the scaler and the model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(knn, 'knn_model.pkl')

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
