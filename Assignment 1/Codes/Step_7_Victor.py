from Step_4_Final import *
print("Step 7")
from sklearn.cluster import KMeans

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)

y_normalized_aligned = y_normalized.reset_index(drop=False)

# Now you can concatenate X_normalized and y_normalized_aligned
df_kmeans = pd.concat([X_normalized, y_normalized_aligned], axis=1)
df_kmeans.set_index('Timestamp', inplace=True)

N_clusters = 5
kmeans = KMeans(n_clusters=N_clusters, random_state=42)

kmeans.fit(X_train)

train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

df_kmeans['cluster'] = kmeans.predict(X_normalized)

# # Plotting
# plt.figure(figsize=(14, 7), dpi = 300)
# plt.scatter(df_kmeans.index, df_kmeans['kalby_active_power'], c=df_kmeans['cluster'], cmap='viridis')  
# plt.colorbar(label='cluster')
# plt.title('K-Means Clustering Visualization')
# plt.xlabel('Timestamp')
# plt.ylabel('Actual Wind Power')
# plt.tight_layout() 
# plt.show()


X_cluster = df_kmeans.drop(['kalby_active_power'], axis=1)
y_cluster = df_kmeans[['kalby_active_power', 'cluster']]

X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, shuffle=False)

# Plotting
plt.figure(figsize=(14, 7), dpi = 300)
plt.scatter(y_test_cluster.index, y_test_cluster['kalby_active_power'], c=y_test_cluster['cluster'], cmap='viridis')  
plt.colorbar(label='cluster')
plt.title('K-Means Clustering Visualization')
plt.xlabel('Timestamp')
plt.ylabel('Actual Wind Power')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

   
predictions = {}

# Loop through the clusters
for cluster_i in range(N_clusters):
    print(cluster_i)
    
    # Training data for the current cluster
    cluster_data_X_train = X_train_cluster[X_train_cluster['cluster'] == cluster_i].drop(columns = 'cluster')
    cluster_data_y_train = y_train_cluster[y_train_cluster['cluster'] == cluster_i]['kalby_active_power']
    # Testing data for the current cluster
    cluster_data_X_test = X_test_cluster[X_test_cluster['cluster'] == cluster_i].drop(columns = 'cluster')
    cluster_data_y_test = y_test_cluster[y_test_cluster['cluster'] == cluster_i]['kalby_active_power']


    if cluster_i in [1,3,4] :     
        theta_closed = np.dot(np.linalg.inv(np.dot(cluster_data_X_train.T, cluster_data_X_train)), 
                               np.dot(cluster_data_X_train.T, cluster_data_y_train))
    
        y_pred_closed_cluster = np.dot(cluster_data_X_test, theta_closed)
        
        predictions[cluster_i] = pd.Series(y_pred_closed_cluster, index=cluster_data_X_test.index)
    
    else :
    
        X_WLS = cluster_data_X_train
        X_query = cluster_data_X_test
        y_WLS = cluster_data_y_train
        
        y_pred_WLS_cluster = weighted_least_squares(X_query, X_WLS, y_WLS, 0.3, tau)
        predictions[cluster_i] = pd.Series(y_pred_WLS_cluster, index=cluster_data_X_test.index)
    
    
    
# Combine all predictions into a single DataFrame
predictions_df = pd.concat(predictions, axis=1)
predictions_df.columns = [f'cluster_{i}' for i in predictions_df.columns]
predictions_df.sort_index(inplace=True)

stacked_df = predictions_df.stack().reset_index()
stacked_df.drop(columns=['level_1'], inplace=True)
stacked_df.sort_values(by='Timestamp', inplace=True)
stacked_df.columns = ['Timestamp', 'Prediction']


mse_k_means = mean_squared_error(y_test, stacked_df['Prediction'])
rmse_k_means = np.sqrt(mse_k_means)
print(f"Test RMSE w/ K-Means: {rmse_k_means:0.10f}")
mae_k_means = mean_absolute_error(y_test, stacked_df['Prediction'])
print(f"Test MAE w/ K-Means: {mae_k_means:0.10f}")


plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_test_cluster.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions NLR
plt.scatter(y_test_cluster.index, stacked_df['Prediction'], color='blue', alpha=0.6, label='Predicted Values - K-Means', s=50)
# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values (K-Means)', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# file_path = 'Figures/Step_4_testing_WLR.png'
# plt.savefig(file_path)

plt.show()


