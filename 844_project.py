import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from apyori import apriori

# Load dataset
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=';')

# Convert wine quality into categorical labels (classification problem)
data['quality_class'] = data['quality'].apply(lambda x: 'Good' if x >= 6 else 'Bad')

# Encode categorical target variable for classification
label_encoder = LabelEncoder()
data['quality_class'] = label_encoder.fit_transform(data['quality_class'])

# Split features and targets
X = data.drop(columns=['quality', 'quality_class'])
y_class = data['quality_class']  # Classification target
y_reg = data['quality']  # Regression target

# Train-Test Split
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Apply PCA (Feature Selection)
pca = PCA(n_components=5)  # Reduce to 5 principal components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

#Define Models
classification_models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest Classifier": RandomForestClassifier(), 
    "Support Vector Classifier": SVC(kernel="rbf"),
}

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor()
}

#Train and Evaluate Classification Models
results_no_fs = {}
results_with_pca = {}
for name, model in classification_models.items():
    model.fit(X_train_scaled, y_train_class)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_class, y_pred)
    results_no_fs[name] = {'Accuracy': acc}
    print(f"{name} (Without Feature Selection): Accuracy = {acc:.4f}")
    print(classification_report(y_test_class, y_pred, target_names=label_encoder.classes_))
    
    model.fit(X_train_pca, y_train_class) 
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test_class, y_pred)
    results_with_pca[name] = {'Accuracy': acc}
    print(f"{name} (With PCA): Accuracy = {acc:.4f}")
    print(classification_report(y_test_class, y_pred, target_names=["Bad", "Good"]))

#Train and Evaluate Regression Models
for name, model in regression_models.items():
    model.fit(X_train_reg_scaled, y_train_reg)
    y_pred = model.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    results_no_fs[name] = {'MSE': mse, 'R^2 Score': r2}
    print(f"{name} (Without Feature Selection): MSE = {mse:.4f}, R^2 Score = {r2:.4f}")
    
    model.fit(X_train_pca, y_train_reg)
    y_pred = model.predict(X_test_pca)
    mse = mean_squared_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)
    results_with_pca[name] = {'MSE': mse, 'R^2 Score': r2}
    print(f"{name} (With PCA): MSE = {mse:.4f}, R^2 Score = {r2:.4f}")
    

# Correlation Matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Compare Performance with and without feature selection
comparison_all_features = pd.DataFrame(results_no_fs)
comparison_selected_features = pd.DataFrame(results_with_pca)

print("\nComparison of Performance with All Features:")
print(comparison_all_features)

print("\nComparison of Performance with Selected Features:")
print(comparison_selected_features)
