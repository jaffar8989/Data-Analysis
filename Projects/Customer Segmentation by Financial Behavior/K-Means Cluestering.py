import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 3.1. Load data
df = pd.read_csv('customer_features.csv')

# 3.2. Preprocessing
features = ['txn_count', 'avg_txn_amount', 'total_amount', 'avg_repayment_days']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3.3. Determine optimal k (Elbow Method)
inertia = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot the Elbow
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# 3.4. Fit K-Means
best_k = 4  # Set after inspecting elbow_plot.png
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 3.5. Attach labels to original data
df['cluster'] = labels

# 3.6. Save final dataset
df.to_csv('customer_segments.csv', index=False)