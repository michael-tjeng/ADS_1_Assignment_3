# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Function to read and clean datasets
def read_and_clean_data(path):
    """
    Reads and cleans the dataset.

    Args:
    path (str): The file path of the dataset.

    Returns:
    DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(path, skiprows=4)
    df = df[['Country Name'] + [str(year) for year in range(1990, 2021)]]
    return df.dropna()

# Function to create a heatmap of the correlation matrix
def plot_correlation_heatmap(df, year):
    """
    Plots a heatmap of the correlation matrix.

    Args:
    df (DataFrame): DataFrame containing the data.
    year (int): The year for which the correlation is analyzed.

    Returns:
    None: The function creates and shows a plot.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix Heatmap for Year {year}')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.show()

# Function for K-Means clustering
def perform_kmeans_clustering(data, n_clusters=3):
    """
    Performs K-Means clustering on the data.

    Args:
    data (DataFrame): DataFrame containing the features for clustering.
    n_clusters (int): The number of clusters.

    Returns:
    tuple: A tuple containing the cluster labels and the centroids.
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    return cluster_labels, centers

# Function to plot clustering results
def plot_clustering_results(data, centers, title='K-Means Clustering'):
    """
    Plots the results of K-Means clustering.

    Args:
    data (DataFrame): DataFrame with clustering data.
    centers (ndarray): Array of cluster centers.
    title (str): Title of the plot.

    Returns:
    None: The function creates and shows a plot.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='gdp_per_capita', y='forest_area', hue='Cluster', 
                    data=data, palette='Set1', s=100)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.75, 
                marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('GDP per Capita (USD)')
    plt.ylabel('Forest Area (% of land area)')
    plt.legend(title='Cluster', loc='upper right')
    plt.savefig('clustering_results.png', dpi=300)
    plt.show()

# Function to calculate silhouette scores
def calculate_silhouette_scores(data, max_clusters):
    """
    Calculates silhouette scores for different numbers of clusters.

    Args:
    data (ndarray): The data to be clustered.
    max_clusters (int): The maximum number of clusters to test.

    Returns:
    dict: A dictionary of silhouette scores for each number of clusters.
    """
    silhouette_scores = {}
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores[n_clusters] = score
    return silhouette_scores

# Function to plot silhouette scores
def plot_silhouette_scores(scores):
    """
    Plots silhouette scores.

    Args:
    scores (dict): Dictionary of silhouette scores.

    Returns:
    None: The function creates and shows a plot.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(2, 11), scores.values())
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 3), 
                 ha='center', va='bottom')
    plt.savefig('silhouette_scores.png', dpi=300)
    plt.show()

# Function to calculate distances to centroids
def calculate_distances_to_centroid(data, centroid, features):
    """
    Calculates the Euclidean distance of each data point to a centroid.

    Args:
    data (DataFrame): The DataFrame with the data points.
    centroid (ndarray): The centroid to which distances are calculated.
    features (list): List of feature columns to use in distance calculation.

    Returns:
    DataFrame: The DataFrame with an additional column for distances.
    """
    data['Distance to Centroid (Original)'] = data.apply(
        lambda row: euclidean(row[features], centroid), axis=1)
    return data.sort_values(by='Distance to Centroid (Original)')

# Main program
def main():
    # Paths to the datasets
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

    # Load and clean data
    dataframes = {key: read_and_clean_data(path) for key, 
                  path in dataset_paths.items()}

    # Ensure all DataFrames have 'Country Name' column
    for df in dataframes.values():
        if 'Country Name' not in df.columns:
            print("Error: 'Country Name' column missing in 1 of the datasets.")
            return

    # Merge datasets
    combined_df = pd.DataFrame()
    for key, df in dataframes.items():
        df_melted = df.melt(id_vars=['Country Name'], var_name='Year', 
                            value_name=key)
        if combined_df.empty:
            combined_df = df_melted
        else:
            combined_df = combined_df.merge(df_melted, 
                                            on=['Country Name', 'Year'], 
                                            how='outer')

    combined_df['Year'] = pd.to_numeric(combined_df['Year'])

    # Select specific year for analysis
    year_for_analysis = 2020
    df_for_correlation = combined_df[combined_df['Year'] == 
                                     year_for_analysis].drop('Year', axis=1)
    df_for_correlation.reset_index(drop=True, inplace=True)

    # Correlation analysis
    plot_correlation_heatmap(df_for_correlation, year_for_analysis)

    # Clustering
    clustering_data = df_for_correlation[['Country Name', 'gdp_per_capita',
                                          'forest_area']].dropna()
    cluster_labels, centers = perform_kmeans_clustering(clustering_data[[
        'gdp_per_capita', 'forest_area']])
    clustering_data['Cluster'] = cluster_labels

    # Plot clustering results
    plot_clustering_results(clustering_data, centers)

    # Silhouette Scores
    silhouette_scores = calculate_silhouette_scores(clustering_data[[
        'gdp_per_capita', 'forest_area']], 10)
    plot_silhouette_scores(silhouette_scores)

    # Calculate distances to centroids
    sorted_distances = {}
    for i in range(3):  # Assuming 3 clusters
        cluster_data = clustering_data[clustering_data['Cluster'] == i]
        sorted_distances[i] = calculate_distances_to_centroid(
            cluster_data, centers[i], ['gdp_per_capita', 'forest_area'])

    # Print sorted distances for each cluster
    for i in range(3):
        print(f"Cluster {i}:\n", sorted_distances[i][[
            'Country Name', 'Distance to Centroid (Original)']].head(), '\n')

if __name__ == "__main__":
    main()
