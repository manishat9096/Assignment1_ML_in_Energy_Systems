from Step_1 import * 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Convert the target lists into pandas Series for consistency
y_G1 = pd.Series(y_G1_values)
y_G2 = pd.Series(y_G2_values)
y_G3 = pd.Series(y_G3_values)

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
# Split data into training and testing sets for each target
X_train_G1, X_test_G1, y_train_G1, y_test_G1 = train_test_split(X_normalized, y_G1, test_size=0.3, random_state=42)
X_train_G2, X_test_G2, y_train_G2, y_test_G2 = train_test_split(X_normalized, y_G2, test_size=0.3, random_state=42)
X_train_G3, X_test_G3, y_train_G3, y_test_G3 = train_test_split(X_normalized, y_G3, test_size=0.3, random_state=42)

# Initialize the classifiers
logistic_clf = LogisticRegression(max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(random_state=42)

# Function to check if there are at least 2 classes in the target
def has_multiple_classes(y):
    return len(y.unique()) > 1

# Function to train and evaluate the classifiers
def evaluate_classifiers(X_train, y_train, X_test, y_test, target_name):
    if has_multiple_classes(y_train):
        # Logistic Regression
        logistic_clf.fit(X_train, y_train)
        y_pred_logistic = logistic_clf.predict(X_test)
        print(f"Classification Report for {target_name} (Logistic Regression):")
        print(classification_report(y_test, y_pred_logistic))
    else:
        y_pred_logistic = []
        print(f"Skipping Logistic Regression for {target_name} - Only one class in training data.")

    if has_multiple_classes(y_train):
        # Random Forest Classifier
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        print(f"Classification Report for {target_name} (Random Forest):")
        print(classification_report(y_test, y_pred_rf))
    else:
        y_pred_rf = []
        print(f"Skipping Random Forest for {target_name} - Only one class in training data.")
        
    return y_pred_logistic, y_pred_rf


# Evaluate models for each target
y_pred_logistic_G1, y_pred_rf_G1 = evaluate_classifiers(X_train_G1, y_train_G1, X_test_G1, y_test_G1, "y_G1")
y_pred_logistic_G2, y_pred_rf_G2 = evaluate_classifiers(X_train_G2, y_train_G2, X_test_G2, y_test_G2, "y_G2")
y_pred_logistic_G3, y_pred_rf_G3 = evaluate_classifiers(X_train_G3, y_train_G3, X_test_G3, y_test_G3, "y_G3")
