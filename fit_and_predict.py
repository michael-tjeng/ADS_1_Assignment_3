import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import errors

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

# Define a function for a quadratic equation
def quadratic_function(x, a, b, c):
    """
    Represents a quadratic function.
    Args:
    x (float or array-like): Independent variable.
    a, b, c (float): Coefficients of the quadratic equation.
    Returns:
    float or array-like: Output of the quadratic function.
    """
    return a * x**2 + b * x + c

# Function to plot both GDP and Forest Area with confidence intervals
def plot_combined_with_confidence(country_name, gdp_data, forest_data, 
                                  params_gdp, pcov_gdp, params_forest, 
                                  pcov_forest, func, years, extended_years, 
                                  extended_years_int):
    """
    Plots GDP and Forest Area with confidence intervals for a given country.
    Args:
    country_name (str): Name of the country.
    gdp_data (array-like): GDP data.
    forest_data (array-like): Forest area data.
    params_gdp (array-like): Parameters for GDP curve fitting.
    pcov_gdp (array-like): Covariance matrix for GDP curve fitting.
    params_forest (array-like): Parameters for forest area curve fitting.
    pcov_forest (array-like): Covariance matrix for forest area curve fitting.
    func (function): Function used for curve fitting.
    years (list): List of years for which data is available.
    extended_years (list): List of extended years as strings for predictions.
    extended_years_int (array-like): Extended years for prediction as integers.
    Returns:
    None: The function creates and shows a plot.
    """
    y_gdp_pred = func(extended_years_int, *params_gdp)
    y_forest_pred = func(extended_years_int, *params_forest)
    sigma_gdp = errors.error_prop(extended_years_int, func, params_gdp, 
                                  pcov_gdp)
    sigma_forest = errors.error_prop(extended_years_int, func, params_forest, 
                                     pcov_forest)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # GDP per Capita Plot
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP per Capita (USD)', color='red')
    ax1.scatter(years, gdp_data, color='red', label='GDP per Capita Data')
    ax1.plot(extended_years, y_gdp_pred, label='Fitted GDP per Capita', 
             color='darkred')
    ax1.fill_between(extended_years, y_gdp_pred - sigma_gdp, 
                     y_gdp_pred + sigma_gdp, color='red', alpha=0.2, 
                     label='Confidence Interval - GDP')
    ax1.tick_params(axis='y', labelcolor='red')

    # Forest Area Plot
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Forest Area (% of land area)', color='green')
    ax2.scatter(years, forest_data, color='green', label='Forest Area Data')
    ax2.plot(extended_years, y_forest_pred, label='Fitted Forest Area', 
             color='darkgreen')
    ax2.fill_between(extended_years, y_forest_pred - sigma_forest, 
                     y_forest_pred + sigma_forest, color='green', alpha=0.2, 
                     label='Confidence Interval - Forest')
    ax2.tick_params(axis='y', labelcolor='green')

    # Set x-axis to show only years ending in 0 and 5 for clarity
    select_years = [year for year in extended_years if year.
                    endswith('0') or year.endswith('5')]
    ax1.set_xticks(select_years)
    ax1.set_xticklabels(select_years, rotation=45)

    # Adjust layout
    fig.tight_layout()
    plt.title(f'Combined Curve Fitting with Confidence Interval for {country_name} (up to 2040)')
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3)
    
    # Save the plot as a PNG file
    plt.savefig(f'{country_name}_combined_plot.png', dpi=300)
    
    plt.show()

# Main program
def main():
    # Paths to the datasets
    dataset_paths = {
        'forest_area': 'API_AG.LND.FRST.ZS_DS2_en_csv_v2_6224694.csv',
        'gdp_per_capita': 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6224630.csv',
    }

    # Load and clean data
    dataframes = {key: read_and_clean_data(path) for key, 
                  path in dataset_paths.items()}

    # Ensure all DataFrames have 'Country Name' column
    for df in dataframes.values():
        if 'Country Name' not in df.columns:
            print("Error: 'Country Name' column missing in 1 of the datasets.")
            return

    # Extract time series data for the selected countries
    years = [str(year) for year in range(1990, 2021)]
    extended_years = [str(year) for year in range(1990, 2041)]
    extended_years_int = np.array(list(map(int, extended_years)))  # For predictions up to 2040

    # Extract and plot the data for each country
    for country in ['Botswana', 'Small states', 'United States']:
        gdp_data = dataframes['gdp_per_capita'][dataframes['gdp_per_capita']['Country Name'] == country][years].T.squeeze()
        forest_data = dataframes['forest_area'][dataframes['forest_area']['Country Name'] == country][years].T.squeeze()
        
        # Re-run the curve fitting to obtain covariance matrix along with the parameters for GDP and Forest Area
        params_gdp, pcov_gdp = curve_fit(quadratic_function, 
                                         np.array(list(map(int, years))), 
                                         gdp_data.values.flatten())
        params_forest, pcov_forest = curve_fit(quadratic_function, 
                                               np.array(list(map(int, years))), 
                                               forest_data.values.flatten())

        # Plot combined chart with confidence intervals
        plot_combined_with_confidence(country, gdp_data.values.flatten(), 
                                      forest_data.values.flatten(), 
                                      params_gdp, pcov_gdp, params_forest, 
                                      pcov_forest, 
                                      quadratic_function, years, 
                                      extended_years, extended_years_int)

if __name__ == "__main__":
    main()
