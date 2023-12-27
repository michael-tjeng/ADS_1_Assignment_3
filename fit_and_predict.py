import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Paths to the datasets (already uploaded by the user)
dataset_paths = {
    'forest_area': 'API_AG.LND.FRST.ZS_DS2_en_csv_v2_6224694.csv',
    'gdp_per_capita': 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6224630.csv',
}

# Load and clean data
dataframes = {}
for key, path in dataset_paths.items():
    df = pd.read_csv(path, skiprows=4)
    df = df[['Country Name'] + [str(year) for year in range(1990, 2021)]]
    df = df.dropna()
    dataframes[key] = df

# Extract time series data for the selected countries
years = [str(year) for year in range(1990, 2021)]
extended_years = [str(year) for year in range(1990, 2041)]
extended_years_int = np.array(list(map(int, extended_years)))  # For predictions up to 2040

# Define a function for a quadratic equation
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

# Function to perform curve fitting and plotting
def fit_and_extend_plot(country_name, gdp_data, forest_data, years, extended_years_int):
    # Prepare the data for curve fitting
    x_data = np.array(list(map(int, years)))  # Convert years to integers for fitting
    y_gdp_data = gdp_data.values.flatten()
    y_forest_data = forest_data.values.flatten()

    # Curve fitting for GDP per Capita
    params_gdp, _ = curve_fit(quadratic_function, x_data, y_gdp_data)
    # Curve fitting for Forest Area
    params_forest, _ = curve_fit(quadratic_function, x_data, y_forest_data)

    # Generate predictions for extended years
    y_gdp_pred = quadratic_function(extended_years_int, *params_gdp)
    y_forest_pred = quadratic_function(extended_years_int, *params_forest)

    # Plotting the original data and the fitted curve
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot GDP per Capita data and fitted curve
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP per Capita', color='red')
    ax1.scatter(years, y_gdp_data, color='red', label='GDP per Capita Data')
    ax1.plot(extended_years, y_gdp_pred, '-', color='red', label='Fitted Curve - GDP per Capita')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot Forest Area data and fitted curve
    ax2 = ax1.twinx()
    ax2.set_ylabel('Forest Area (% of land area)', color='green')
    ax2.scatter(years, y_forest_data, color='green', label='Forest Area Data')
    ax2.plot(extended_years, y_forest_pred, '-', color='green', label='Fitted Curve - Forest Area')
    ax2.tick_params(axis='y', labelcolor='green')

    # Set x-axis to show only years ending in 0 and 5 for clarity
    select_years = [year for year in extended_years if year.endswith('0') or year.endswith('5')]
    ax1.set_xticks(select_years)
    ax1.set_xticklabels(select_years, rotation=45)  # Rotate the x-axis labels for better readability

    fig.tight_layout()  # Adjust layout
    plt.title(f'Curve Fitting and Prediction for {country_name} (up to 2040)')
    fig.legend(loc="center left", bbox_to_anchor=(0.05, -0.01), ncol=4)
    plt.show()

    return params_gdp, params_forest

# Extract the data for Botswana
gdp_botswana = dataframes['gdp_per_capita'][dataframes['gdp_per_capita']['Country Name'] == 'Botswana'][years].T.squeeze()
forest_botswana = dataframes['forest_area'][dataframes['forest_area']['Country Name'] == 'Botswana'][years].T.squeeze()

# Extract the data for Brazil
gdp_brazil = dataframes['gdp_per_capita'][dataframes['gdp_per_capita']['Country Name'] == 'Brazil'][years].T.squeeze()
forest_brazil = dataframes['forest_area'][dataframes['forest_area']['Country Name'] == 'Brazil'][years].T.squeeze()

# Extract the data for Sweden
gdp_sweden = dataframes['gdp_per_capita'][dataframes['gdp_per_capita']['Country Name'] == 'Sweden'][years].T.squeeze()
forest_sweden = dataframes['forest_area'][dataframes['forest_area']['Country Name'] == 'Sweden'][years].T.squeeze()

# Perform curve fitting and plotting for Botswana
params_botswana_gdp, params_botswana_forest = fit_and_extend_plot('Botswana', gdp_botswana, forest_botswana, years, extended_years_int)

# Perform curve fitting and plotting for Brazil
params_brazil_gdp, params_brazil_forest = fit_and_extend_plot('Brazil', gdp_brazil, forest_brazil, years, extended_years_int)

# Perform curve fitting and plotting for Sweden
params_sweden_gdp, params_sweden_forest = fit_and_extend_plot('Sweden', gdp_sweden, forest_sweden, years, extended_years_int)