
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


import pandas as pd

######## Uploading and treating the features ##########
features = pd.read_csv('Datasets/Dataset.csv')

# We ant to create dataframes that contain for each hour of the day, the data from each sample.
# So, we create 24 dataframes containing 1000 samples for an hour.
# Then, we split the data in training, validation, test sets

# Select the different features (one table for each hour)
num_dataframes = 24

# List to store DataFrames
dataframes = []

# Create DataFrames starting from each row and selecting every 24th row after it
for start_row in range(num_dataframes):
    df = features.iloc[start_row::24]
    dataframes.append(df)

# Normalize the dataframes
scaler = MinMaxScaler()

# List to store the normalized dataframes
X_train = []
X_val = []
X_test = []

# The training set is composed of the first 60% of samples created, the validation of 20% and testing of 20%
split_index_train = int(0.6 * len(dataframes[0]))
split_index_val = int(0.8 * len(dataframes[0]))


for X in dataframes:
    ## Fit and transform the selected columns
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=features.columns)
    X_train.append(X_normalized[:split_index_train])
    X_val.append(X_normalized[split_index_train: split_index_val])
    X_test.append(X_normalized[split_index_val:])


######## Uploading and treating the target ###########
target = pd.read_csv('Datasets/Target.csv')

# List to store DataFrames
hourly_target = []

# Create DataFrames starting from each row and selecting every 24th row after it
for start_row in range(num_dataframes):
    df = target.iloc[start_row::24]
    hourly_target.append(df)

# List to store the normalized dataframes
y_G1 = []
y_G2 = []
y_G3 = []

for Y in hourly_target:
    ## Fit and transform the selected columns
    y_G1.append(Y['G1'])
    y_G2.append(Y['G2'])
    y_G3.append(Y['G3'])

#some values to check
# print(y_G1.value_counts())
# print(y_G2.value_counts())
# print(y_G3.value_counts())


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, X_normalized, y):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluation of the model
    # print(f"Model: {model_name}")
    # print("Accuracy:", accuracy_score(y_test, y_pred))


    # ROC-AUC Score - range 0 to 1 - higher score better classifier
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]


    # All evaluation metrics added to metrics list
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
    }

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        metrics['ROC-AUC'] = roc_auc

    # Confusion Matrix - counts of true positives, false positives, true negatives, and false negatives
    cm = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:\n", cm)

    return model, y_pred, metrics

metrics_dict = {}

def all_classifiers(X_train, X_val, y, split_index_train, split_index_val, label, n_neighbors, val_test, h):
    global metrics_dict
    if len(y.unique()) > 1:  # Only proceed if there are more than two unique values
        
        if val_test == 'VAL':
            y_train = y[:split_index_train]
            y_val = y[split_index_train:split_index_val]
        
        if val_test == 'TEST':
            y_train = y[:split_index_train]
            y_val = y[split_index_val:]
        metrics_list = []
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "SVM Linear": SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
            "SVM Gaussian": SVC(kernel='rbf', gamma='scale', random_state=42, class_weight='balanced', probability=True),
            "KNeighbors": KNeighborsClassifier(n_neighbors)
        }

        for model_name, model in models.items():
            trained_model, y_pred, metrics = train_and_evaluate(
                model, model_name, X_train, X_val, y_train, y_val, X_train, y
            )

            # Create an identifier for the model
            model_key = f"{model_name}_{label}_{h}"

            # Save metrics under the model identifier
            metrics_dict[model_key] = metrics

# Process each target for each hour

n_neighbors = 10


for h in range(len(y_G1)):
    
    val_test = 'VAL'
    
    metrics_df_G1 = all_classifiers(X_train[h], X_val[h], y_G1[h], split_index_train, split_index_val, "G1", n_neighbors, val_test, h)
    metrics_df_G2 = all_classifiers(X_train[h], X_val[h], y_G2[h], split_index_train, split_index_val, "G2", n_neighbors, val_test, h)
    metrics_df_G3 = all_classifiers(X_train[h], X_val[h], y_G3[h], split_index_train, split_index_val, "G3", n_neighbors, val_test, h)


    # val_test = 'TEST'
    
    # all_classifiers(X_train[h], X_test[h], y_G1[h], split_index_train, split_index_val, "G1", n_neighbors, val_test, h)
    # all_classifiers(X_train[h], X_test[h], y_G2[h], split_index_train, split_index_val, "G2", n_neighbors, val_test, h)
    # all_classifiers(X_train[h], X_test[h], y_G3[h], split_index_train, split_index_val, "G3", n_neighbors, val_test, h)

metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
metrics_df.index.name = "Model_Label_Hour"

