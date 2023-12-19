import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the datasets uploaded by the user
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

# Initialize a dictionary to hold the dataframes
dataframes = {}

# Load the datasets into pandas dataframes
for key, path in dataset_paths.items():
    dataframes[key] = pd.read_csv(path, skiprows=4)

# Function to preprocess and perform EDA on each dataframe
def preprocess_and_eda(df):
    # Dropping columns with no relevance to analysis
    df.drop(columns=['Indicator Code', 'Country Code', 'Unnamed: 67'], inplace=True, errors='ignore')
    
    # Dropping rows where all data entries are NaN (if any)
    df.dropna(axis=0, how='all', subset=df.columns[4:], inplace=True)
    
    # EDA can include more steps as needed, such as:
    # - Descriptive statistics
    # - Checking for and imputing missing values
    # - Identifying outliers
    # - Normalizing or scaling data
    # - Visualizing distributions of variables
    # For now, we'll calculate and return the descriptive statistics.
    eda_results = df.describe(include='all').T
    return eda_results

# Perform EDA for each indicator
eda_results = {}
for key, df in dataframes.items():
    eda_results[key] = preprocess_and_eda(df)

print(eda_results['agricultural_land'].head())  # Display an example of EDA results for one indicator

# Function to find the most recent common year with data for most of the 222 countries for each indicator
def find_recent_common_year_for_countries(dataframes_dict, common_countries):
    recent_common_year = None
    for year in range(2022, 1959, -1):  # From 2022 backwards to 1960
        year_str = str(year)
        year_data_complete = True
        for df in dataframes_dict.values():
            # Check if the year column exists and if data is available for most of the common countries
            if year_str not in df.columns or df[df['Country Name'].isin(common_countries)][year_str].isna().sum() > len(common_countries) * 0.1:
                year_data_complete = False
                break
        if year_data_complete:
            recent_common_year = year_str
            break
    return recent_common_year

# Approach 3: Focus on a subset of countries with complete data for the most recent year across all indicators
# We will find the most recent year for each indicator where data is available for all countries
# and then identify the common countries with complete data for all indicators in those years.

# Function to find the most recent year with complete data for each indicator
def find_most_recent_year(dataframe):
    # Getting a list of years for which the dataframe has complete data
    years_with_complete_data = dataframe.columns[~dataframe.isnull().any()].tolist()[4:]
    return max(years_with_complete_data) if years_with_complete_data else None

# Find the most recent year with complete data for each indicator
recent_years = {key: find_most_recent_year(df) for key, df in dataframes.items()}

# Filter each dataframe to include only data from the most recent year identified
filtered_dataframes = {}
for key, year in recent_years.items():
    if year:
        filtered_dataframes[key] = dataframes[key][['Country Name', year]]

# Identify common countries with complete data across all indicators
common_countries = None
for df in filtered_dataframes.values():
    countries_with_data = set(df['Country Name'].dropna())
    if common_countries is None:
        common_countries = countries_with_data
    else:
        common_countries = common_countries.intersection(countries_with_data)

# Filter each dataframe to include only the common countries
for key in filtered_dataframes:
    filtered_dataframes[key] = filtered_dataframes[key][filtered_dataframes[key]['Country Name'].isin(common_countries)]

# Merge the filtered dataframes for correlation analysis
combined_filtered_data = pd.DataFrame()
for key, df in filtered_dataframes.items():
    df = df.rename(columns={df.columns[1]: key})
    if combined_filtered_data.empty:
        combined_filtered_data = df
    else:
        combined_filtered_data = combined_filtered_data.merge(df, on='Country Name', how='inner')

# Calculate the correlation matrix for the combined filtered dataframe
correlation_matrix = combined_filtered_data.set_index('Country Name').corr()

correlation_matrix, recent_years, len(common_countries)  # Display the correlation matrix, the years used, and the number of common countries

# Find the most recent common year with data for most countries
recent_common_year_for_countries = find_recent_common_year_for_countries(dataframes, common_countries)

# If a common year is found, extract data for that year from each indicator
if recent_common_year_for_countries:
    # Initialize a new dataframe to hold all the data for the common year
    combined_common_year_data = pd.DataFrame()

    # Extract data for the common year from each indicator
    for key, df in dataframes.items():
        temp_df = df[['Country Name', recent_common_year_for_countries]].copy()
        temp_df.rename(columns={recent_common_year_for_countries: key}, inplace=True)
        if combined_common_year_data.empty:
            combined_common_year_data = temp_df
        else:
            combined_common_year_data = combined_common_year_data.merge(temp_df, on='Country Name', how='inner')

    # Calculate the correlation matrix for the combined data
    correlation_matrix_common_year = combined_common_year_data.set_index('Country Name').corr()
else:
    correlation_matrix_common_year = "No recent common year with sufficient data found."

print(correlation_matrix_common_year) 
print(recent_common_year_for_countries)  # Display the correlation matrix and the common year used

# Creating the heatmap with tilted x-axis labels for better readability
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_common_year, annot=True, fmt=".2f", cmap='coolwarm')
# Adding title and labels
plt.title('Correlation Matrix Heatmap for the Year 2020')
plt.xlabel('Indicators')
plt.ylabel('Indicators')
# Tilting the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# Show the heatmap
plt.show()
