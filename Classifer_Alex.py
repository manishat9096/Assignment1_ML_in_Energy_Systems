
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

Xpd = pd.read_csv('Datasets/Dataset.csv')
y = pd.read_csv('Datasets/Target.csv')

#normalize the X dataset
# scaler = MinMaxScaler()
scaler = StandardScaler()
X_normalized = scaler.fit_transform(Xpd)
y_G1 = pd.Series(y['G1'])
y_G2 = pd.Series(y['G2'])
y_G3 = pd.Series(y['G3'])

#stratify ensures ratio of classes is same for subsets
X_tr_G1, X_test_G1, y_tr_G1, y_test_G1 = train_test_split(X_normalized, y_G1, test_size= 0.2, random_state=42, stratify=y_G1)
X_tr_G2, X_test_G2, y_tr_G2, y_test_G2 = train_test_split(X_normalized, y_G2, test_size= 0.2, random_state=42, stratify=y_G2)
X_tr_G3, X_test_G3, y_tr_G3, y_test_G3 = train_test_split(X_normalized, y_G3, test_size= 0.2, random_state=42, stratify=y_G3)

X_train_G1, X_val_G1, y_train_G1, y_val_G1 = train_test_split(X_tr_G1, y_tr_G1, test_size= 0.2, random_state=42, stratify=y_tr_G1)
X_train_G2, X_val_G2, y_train_G2, y_val_G2 = train_test_split(X_tr_G2, y_tr_G2, test_size= 0.2, random_state=42, stratify=y_tr_G2)
X_train_G3, X_val_G3, y_train_G3, y_val_G3 = train_test_split(X_tr_G3, y_tr_G3, test_size= 0.2, random_state=42, stratify=y_tr_G3)


# def hyperparameter_tuning(X_train, X_test, y_train, y_test, X_normalized):
#     # Define parameter grid
#     param_grid = {
#         'C': [0.1, 1, 10, 100, 1000],
#         'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
#         'kernel': ['rbf'],
#         'class_weight': [None, 'balanced']
#     }

#     svm_test = SVC(random_state=42)

#     # Grid search
#     grid_search = GridSearchCV( estimator=svm_test, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     print("Best parameters:", grid_search.best_params_)

#     best_svm = grid_search.best_estimator_

#     y_pred = best_svm.predict(X_test)

#     print("Accuracy", accuracy_score(y_test, y_pred))
#     print("Classification Report\n",classification_report(y_test, y_pred))

#     #print scores
#     scores = cross_val_score(best_svm, X_normalized, y_G1, cv=5, scoring='accuracy')
#     print("Cross-validation scores:", scores)
#     print("Mean cross-validation score:", scores.mean())
    
#     return y_pred


# y_pred_G1 = hyperparameter_tuning(X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized)
# y_pred_G2= hyperparameter_tuning(X_train_G2, X_test_G2, y_train_G2, y_test_G2, X_normalized)
# y_pred_G3 = hyperparameter_tuning(X_train_G3, X_test_G3, y_train_G3, y_test_G3, X_normalized)



#some values to check
# print(y_G1.value_counts())
# print(y_G2.value_counts())
# print(y_G3.value_counts())


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, X_normalized, y):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluation of the model
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report\n", classification_report(y_test, y_pred))

    # Cross-validation scores of the model
    scores = cross_val_score(model, X_normalized, y, cv=skf, scoring='accuracy')
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

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
    metrics['Cross-Validation Score'] = scores.mean()

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        metrics['ROC-AUC'] = roc_auc
        print("ROC-AUC Score:", roc_auc)

    # Confusion Matrix - counts of true positives, false positives, true negatives, and false negatives
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return model, y_pred, metrics

if __name__ == '__main__':
    metrics_list = []

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model, y_pred_lr, metrics_lr = train_and_evaluate(
        lr_model, 'Logistic Regression', X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized, y_G1
    )
    metrics_list.append(metrics_lr)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model, y_pred_rf, metrics_rf = train_and_evaluate(
        rf_model, 'Random Forest', X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized, y_G1
    )
    metrics_list.append(metrics_rf)

    #SVM linear model
    svmlinear = SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True)
    svmlinear, y_pred_svm_linear, metrics_svml = train_and_evaluate(
        svmlinear, 'SVM Linear', X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized, y_G1
    )
    metrics_list.append(metrics_svml)

    #SVM gaussian model
    svmgaussian = SVC(kernel='rbf', gamma= 'scale', random_state=42, class_weight='balanced', probability=True)
    svmgaussian, y_pred_svm_gaussian, metrics_svmg = train_and_evaluate(
        svmgaussian, 'SVM Gaussian', X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized, y_G1
    )
    metrics_list.append(metrics_svmg)

    #KNeighbours Model
    Kneigh = KNeighborsClassifier(n_neighbors=3)
    Kneigh, y_pred_knn, metrics_knn = train_and_evaluate(
        Kneigh, 'KNeighbours', X_train_G1, X_test_G1, y_train_G1, y_test_G1, X_normalized, y_G1
    )
    metrics_list.append(metrics_knn)

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df)

print('End Step 3 M')
    

