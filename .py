
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
from matplotlib import pyplot as plt

# Specify the file path of your CSV file
file_path = "data/AZWatch_subscribers.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Separate predictor variables from class label
X = df.drop(['subscriber_id','subscription_status'], axis=1)
y = df.subscription_status

# Split intro training and test sets (20% test)
X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=.2, random_state=42)


# Create a copy for encoding
X_encoded = X.copy()

# Identify categorical columns
categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns

# Apply Label Encoding
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le


# Initialize StandardScaler
scaler = StandardScaler()

# Fit on full encoded X, then transform train/test splits
X_scaled = scaler.fit_transform(X_encoded)

# Use index alignment to match scaled features with train/test splits
X_train_scaled = scaler.transform(X_encoded.loc[X_train.index])
X_test_scaled = scaler.transform(X_encoded.loc[X_test.index])

# Logistic Regression
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train_scaled, y_train)
acc1 = model1.score(X_test_scaled, y_test)

# Decision Tree
model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train_scaled, y_train)
acc2 = model2.score(X_test_scaled, y_test)

# Random Forest
model3 = RandomForestClassifier(random_state=42)
model3.fit(X_train_scaled, y_train)
acc3 = model3.score(X_test_scaled, y_test)

# Store best score
score = max(acc1, acc2, acc3)
print(f"Logistic Regression Accuracy: {acc1:.2%}")
print(f"Decision Tree Accuracy: {acc2:.2%}")
print(f"Random Forest Accuracy: {acc3:.2%}")
print(f"Best Model Accuracy (score): {score:.2%}")

y_pred = model1.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model1.classes_)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# Clustering: select numerical features
segmentation = X_encoded.select_dtypes(include='number').copy()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster_id'] = kmeans.fit_predict(segmentation)

# Add cluster_id to segmentation DataFrame for analysis
segmentation['cluster_id'] = df['cluster_id']

# Analyze cluster averages
analysis = segmentation.groupby('cluster_id')[['engagement_time', 'engagement_frequency']].mean().round(0)
print("\nCluster Analysis (Rounded Averages):")
print(analysis)
