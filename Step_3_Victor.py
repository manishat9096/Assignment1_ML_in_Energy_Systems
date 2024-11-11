from Step_1 import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

# Convert the target lists into pandas Series for consistency
y_G1 = pd.Series(y_G1_values)
y_G2 = pd.Series(y_G2_values)
y_G3 = pd.Series(y_G3_values)

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split data into training and testing sets for each target
split_index = int(0.8 * len(X_normalized))

X_train = X_normalized[:split_index]
X_test = X_normalized[split_index:]


"""
######## Uploading and treating the features ##########
features = pd.read_csv('Dataset.csv')

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
target = pd.read_csv('Target.csv')

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
"""

##################################################

# Function to check and process each target
def process_target(X_train, X_test, split_index, y, label, kernel_SVM):
    if len(y.unique()) > 1:  # Only proceed if there are more than two unique values

        y_train = y[:split_index]
        y_test = y[split_index:]

        # KNeighbors Classifier
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        predictions_KN = neigh.predict(X_test)

        evaluate_model(y_test, predictions_KN, f"{label} Classifier - K-Neighbours")

        # SVM
        clf = svm.SVC(kernel=kernel_SVM)
        clf.fit(X_train, y_train)
        predictions_SVM = clf.predict(X_test)

        evaluate_model(
            y_test, predictions_SVM, f"{label} Classifier - SVM ({kernel_SVM})"
        )

        # # SVM with PCA for 2D visualization
        # X_reduced = PCA(n_components=2).fit_transform(X_train)
        # fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        # plot_training_data_with_decision_boundary(kernel_SVM, X_reduced, y_train, ax=ax)
        # plt.show()


# Plot function for predictions and actual values
def plot_predictions(y_test, predictions, title):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(range(len(y_test)), y_test, label="Actual", color="b", alpha=0.6, s=40)
    plt.scatter(
        range(len(predictions)),
        predictions,
        label="Predicted",
        color="r",
        alpha=0.6,
        s=40,
        marker="x",
    )
    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Class Label", fontsize=14)
    plt.title(f"{title} - Actual vs Predicted", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()


# Evaluate model with classification report and plot
def evaluate_model(y_test, predictions, title):
    plot_predictions(y_test, predictions, f"{title}")
    report = classification_report(y_test, predictions)
    print(f"Classification Report for {title}:\n")
    print(report)


# SVM plot function with decision boundary
def plot_training_data_with_decision_boundary(
    kernel, X_reduced, y, ax=None, long_title=True, support_vectors=True
):
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X_reduced, y)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    common_params = {"estimator": clf, "X": X_reduced, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Unit Commitment")
    ax.set_xlabel("PCA Component 1", fontsize=14)
    ax.set_ylabel("PCA Component 2", fontsize=14)
    ax.set_title(f"Decision Boundaries of {kernel} Kernel in SVC with PCA", fontsize=16)
    ax.grid(True)


# Process each target
# kernel_SVM = "linear"
kernel_SVM = "poly"
# kernel_SVM = "rbf"
process_target(X_train, X_test, split_index, y_G1, "G1", kernel_SVM)
process_target(X_train, X_test, split_index, y_G2, "G2", kernel_SVM)
process_target(X_train, X_test, split_index, y_G3, "G3", kernel_SVM)
