import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Paths to the datasets (update these paths according to your file locations)
dataset_paths = {
    'agricultural_land': 'API_AG.LND.AGRI.ZS_DS2_en_csv_v2_6225048.csv',
    'freshwater_withdrawals': 'API_ER.H2O.FWTL.ZS_DS2_en_csv_v2_6229258.csv',
    'co2_emissions': 'API_EN.ATM.CO2E.PC_DS2_en_csv_v2_6225292.csv',
    'forest_area': 'API_AG.LND.FRST.ZS_DS2_en_csv_v2_6224694.csv',
    'gdp_per_capita': 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6224630.csv',
    'under5_mortality': 'API_SH.DYN.MORT_DS2_en_csv_v2_6225034.csv',
    'renewable_energy': 'API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_6224801.csv',
    'urban_population': 'API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_6227277.csv'
}

# Loading and cleaning data
dataframes = {}
for key, path in dataset_paths.items():
    df = pd.read_csv(path, skiprows=4)
    df = df[['Country Name'] + [str(year) for year in range(1990, 2021)]]
    df = df.dropna()
    dataframes[key] = df

# Merging datasets
combined_df = pd.DataFrame()
for key, df in dataframes.items():
    df_melted = df.melt(id_vars=['Country Name'], var_name='Year', value_name=key)
    if combined_df.empty:
        combined_df = df_melted
    else:
        combined_df = combined_df.merge(df_melted, on=['Country Name', 'Year'])

combined_df['Year'] = pd.to_numeric(combined_df['Year'])

# Selecting a specific year for correlation analysis, e.g., 2020
year_for_analysis = 2020
df_for_correlation = combined_df[combined_df['Year'] == year_for_analysis].drop('Year', axis=1)
df_for_correlation.reset_index(drop=True, inplace=True)

# Calculating the correlation matrix
correlation_matrix = df_for_correlation.corr()

# Creating a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap for Year ' + str(year_for_analysis))
plt.xticks(rotation=45, ha='right')
plt.show()

# Clustering
clustering_data = df_for_correlation[['Country Name', 'gdp_per_capita', 'forest_area']].dropna()
features_for_scaling = clustering_data[['gdp_per_capita', 'forest_area']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_scaling)

kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)
clustering_data['Cluster'] = cluster_labels

# Transform the centroids back to the original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

print(centers)

# Print the centroids
print("Centroids (Cluster 0, 1, 2):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: GDP per capita = {center[0]:.2f}, Forest Area = {center[1]:.2f}")

# Plotting the clustering results
plt.figure(figsize=(12, 8))
sns.scatterplot(x='gdp_per_capita', y='forest_area', hue='Cluster', data=clustering_data, palette='Set1', s=100)
plt.title('K-Means Clustering on GDP per Capita and Forest Area')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Forest Area (% of land area)')

# Plotting cluster centers (centroids)
# Transform the centroids back to the original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, marker='X', label='Centroids')

plt.legend(title='Cluster', loc='upper right')
plt.show()

# Silhouette Scores
def calculate_silhouette_scores(data, max_clusters):
    silhouette_scores = {}
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores[n_clusters] = score
    return silhouette_scores

silhouette_scores = calculate_silhouette_scores(features_scaled, 10)

plt.figure(figsize=(10, 6))
bars = plt.bar(range(2, 11), silhouette_scores.values())
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Numbers of Clusters')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 3), ha='center', va='bottom')
plt.show()

print(clustering_data, '\n')

# Function to calculate distances to centroid in original feature space
def calculate_original_distances_to_centroid(cluster_data, centroid):
    cluster_data['Distance to Centroid (Original)'] = cluster_data.apply(
        lambda row: euclidean(row[['gdp_per_capita', 'forest_area']], centroid), axis=1)
    return cluster_data.sort_values(by='Distance to Centroid (Original)')

# Apply the function to each cluster and store results
sorted_original_distances = {}
for i in range(3):  # Assuming there are 3 clusters
    cluster_data = clustering_data[clustering_data['Cluster'] == i]
    # Use the original scale centroids
    original_centroid = centers[i]
    sorted_original_distances[i] = calculate_original_distances_to_centroid(cluster_data, original_centroid)

# Example: Print sorted distances for Cluster 0 in original feature space
print(sorted_original_distances[0][['Country Name', 'Distance to Centroid (Original)']].head(), '\n')
print(sorted_original_distances[1][['Country Name', 'Distance to Centroid (Original)']].head(), '\n')
print(sorted_original_distances[2][['Country Name', 'Distance to Centroid (Original)']].head(), '\n')