from Step_2_Final import *
from Step_4_Final import weighted_least_squares
print("Step 7")
from sklearn.cluster import KMeans
import itertools
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)

y_normalized_aligned = y_normalized.reset_index(drop=False)

# Now you can concatenate X_normalized and y_normalized_aligned
df_kmeans = pd.concat([X_normalized, y_normalized_aligned], axis=1)
df_kmeans.set_index('Timestamp', inplace=True)

N_clusters = 5
kmeans = KMeans(n_clusters=N_clusters, random_state=42)

kmeans.fit(X_train)
test_clusters = kmeans.predict(X_test)

df_kmeans['cluster'] = kmeans.predict(X_normalized)

# Plotting
plt.figure(figsize=(16, 7), dpi = 300)
plt.scatter(df_kmeans.loc[y_test.index].index, y_test, c=test_clusters, cmap='viridis')  
plt.colorbar(label='Clusters')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
file_path = 'Figures/Step_7_clustering.png'
plt.savefig(file_path)
plt.show()

X_cluster = df_kmeans.drop(['kalby_active_power'], axis=1)
y_cluster = df_kmeans[['kalby_active_power', 'cluster']]

X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, shuffle=False)


def evaluate_combination(combination, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster):
    predictions = {}
    total_mse = 0
    
    for cluster_i in range(N_clusters):
        cluster_data_X_train = X_train_cluster[X_train_cluster['cluster'] == cluster_i].drop(columns='cluster')
        cluster_data_y_train = y_train_cluster[y_train_cluster['cluster'] == cluster_i]['kalby_active_power']
        cluster_data_X_test = X_test_cluster[X_test_cluster['cluster'] == cluster_i].drop(columns='cluster')
        cluster_data_y_test = y_test_cluster[y_test_cluster['cluster'] == cluster_i]['kalby_active_power']
        
        if combination[cluster_i] == 0:  # Apply Linear Regression (LR)
            theta_closed = np.dot(np.linalg.inv(np.dot(cluster_data_X_train.T, cluster_data_X_train)),
                                  np.dot(cluster_data_X_train.T, cluster_data_y_train))
            y_pred_cluster = np.dot(cluster_data_X_test, theta_closed)
        else:  # Apply Weighted Least Squares (WLS)
            y_pred_cluster = weighted_least_squares(cluster_data_X_test, cluster_data_X_train, cluster_data_y_train, 0.26666)
        
        predictions[cluster_i] = pd.Series(y_pred_cluster, index=cluster_data_X_test.index)
    
    predictions_df = pd.concat(predictions, axis=1)
    predictions_df.columns = [f'cluster_{i}' for i in predictions_df.columns]
    predictions_df.sort_index(inplace=True)

    stacked_df = predictions_df.stack().reset_index()
    stacked_df.drop(columns=['level_1'], inplace=True)
    stacked_df.sort_values(by='Timestamp', inplace=True)
    stacked_df.columns = ['Timestamp', 'Prediction']


    mse_k_means = mean_squared_error(y_test, stacked_df['Prediction'])
    
    return mse_k_means, stacked_df

# Generate all possible combinations of LR (0) and WLS (1)
combinations = list(itertools.product([0, 1], repeat=N_clusters))

best_combination = None
best_rmse = float('inf')
best_pred = 0

for comb in combinations:
    print(f"Evaluating combination: {comb}")
    mse = evaluate_combination(comb, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)[0]
    pred = evaluate_combination(comb, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster)[1]
    rmse_k_means = np.sqrt(mse)
    print(f"Test RMSE w/ K-Means: {rmse_k_means:0.10f}")
    
    if rmse_k_means < best_rmse:
        best_rmse = rmse_k_means
        best_combination = comb
        best_pred = pred
        
print(f"Best combination: {best_combination} with MSE: {best_rmse}")

plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions NLR
plt.scatter(y_test.index, best_pred['Prediction'], color='blue', alpha=0.6, label='Predicted Values - K-Means', s=50)
# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values (best K-Means)', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
file_path = 'Figures/Step_7_testing.png'
plt.savefig(file_path)
plt.show()




