import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

COLUMN_NAMES_KEY = {
    "ts": "timestamp",
    "t": "temperature",
    "w": "wind_speed",
    "s": "ghi",
    "d": "dhi",
    "n": "dni",
}

SOLAR_CAPACITY_KW = 8
SOLAR_EFFICIENCY_FACTOR = 0.8
INDOOR_TEMPERATURE = 20
HEAT_LOSS_KW_DEG_C = 0.10

HOT_WATER_USAGE = 100  # litres per day
HOT_WATER_COP = 2.5  # Coefficient of performance of the heat pump for hot water
HOT_WATER_KWH_PER_LITRE = 0.035  # kWh per litre of hot water
DAILY_HOT_WATER_USAGE_KWH = HOT_WATER_USAGE * HOT_WATER_KWH_PER_LITRE / HOT_WATER_COP

OTHER_ELECTRICITY_USAGE_DAY = 10  # kWh

VEHICLE_ANNUAL_MILEAGE = 10000  # km
VEHICLE_CONSUMPTION_KWH_PER_MILE = 0.3  # kWh per mile
VEHICLE_HOURLY_CONSUMPTION = VEHICLE_CONSUMPTION_KWH_PER_MILE * VEHICLE_ANNUAL_MILEAGE / 365 / 24

def create_scaled_load_profile(total_daily_kwh):
    """
    Creates a crude 24-hour load profile for residential electricity demand and scales it
    to match a given total daily kWh usage.
    
    Parameters:
    total_daily_kwh (float): The total electricity consumption in kWh for the day.
    
    Returns:
    np.array: A 24-element array representing electricity demand in kW for each hour of the day.
    """
    # Crude load profile based on relative demands during different times of the day
    load_profile = np.array([
        0.4, 0.3, 0.3, 0.3, 0.3, 0.4,  # Night (12 AM to 6 AM)
        0.8, 1.0, 0.9,                 # Morning peak (7 AM to 9 AM)
        0.7, 0.6, 0.6, 0.7, 0.8, 1.0, 1.1, 1.3,  # Daytime (9 AM to 4 PM)
        2.0, 2.2, 2.3, 2.0, 1.8, 1.4,  # Evening peak (5 PM to 9 PM)
        1.0, 0.8, 0.6, 0.4             # Late evening (10 PM to midnight)
    ])
    
    # Calculate the sum of the crude load profile (in kWh)
    total_crude_kwh = np.sum(load_profile)
    
    # Scale the load profile to match the total daily kWh consumption
    scale_factor = total_daily_kwh / total_crude_kwh
    scaled_load_profile = load_profile * scale_factor
    
    return scaled_load_profile

DAILY_LOAD_PROFILE = create_scaled_load_profile(OTHER_ELECTRICITY_USAGE_DAY)

def open_data():
    df = pd.read_json("weather_history.json")
    # Rename columns
    df.rename(columns=COLUMN_NAMES_KEY, inplace=True)
    df.set_index("timestamp", inplace=True)
    df.index = pd.to_datetime(df.index, unit="ms")
    return df


def estimate_solar_power(
    df,
    system_capacity_kw=SOLAR_CAPACITY_KW,
    panel_efficiency=0.22,
    system_loss_factor=0.95,
):
    """
    Estimates the power output of a solar PV system based on solar radiation data and panel efficiency.

    Parameters:
    ghi (float): Global Horizontal Irradiance in W/m²
    dni (float): Direct Normal Irradiance in W/m² (not directly used in this basic model)
    dhi (float): Diffuse Horizontal Irradiance in W/m² (not directly used in this basic model)
    system_capacity_kw (float): The total capacity of the solar PV system in kW (default: 8 kW)
    panel_efficiency (float): Efficiency of the solar panels (default: 0.22 for 22%)
    system_loss_factor (float): Losses due to inverter, wiring, temperature, shading, etc. (default: 0.85 for 15% losses)

    Returns:
    float: Estimated power output in kW.
    """
    # Convert GHI to energy per square meter in kW (GHI is in W/m², so divide by 1000)
    power_output_per_m2 = df["ghi"] / 1000 * panel_efficiency

    # Total output is scaled by system capacity and system losses
    df["solar"] = system_capacity_kw * power_output_per_m2 * system_loss_factor

    return df


def estimate_heat_demand(df, indoor_temp=INDOOR_TEMPERATURE, heat_loss_coeff=HEAT_LOSS_KW_DEG_C, solar_efficiency=SOLAR_EFFICIENCY_FACTOR):
    """
    Estimate the heat demand for a semi-detached house based on outdoor temperature and solar gains.

    Parameters:
    outdoor_temp (float): Outdoor temperature in °C.
    indoor_temp (float): Desired indoor temperature in °C (default is 20°C).
    heat_loss_coeff (float): Heat loss coefficient in KW/°C (default is 0.2 KW/°C).
    solar_efficiency (float): Efficiency factor for converting solar radiation to heat gains (default is 0.8).

    Returns:
    float: Estimated heat demand in kilowatt-hours (kWh).
    """
    # Calculate the temperature difference
    outdoor_temp = df['temperature']
    
    temp_diff = indoor_temp - outdoor_temp

    # Heat demand in kW
    heat_demand_kwh = heat_loss_coeff * temp_diff

    # Calculate solar gains (assuming GHI is the relevant measure for solar gains)
    solar_gains = df['ghi'] / 1000 * solar_efficiency

    # Adjust heat demand by subtracting solar gains
    net_heat_demand_kwh = heat_demand_kwh - solar_gains

    # Ensure that net heat demand does not go below zero
    net_heat_demand_kwh[net_heat_demand_kwh < 0] = 0

    # Estimate the COP of the heat pump based on the outdoor temperature
    cop = pd.Series(index=outdoor_temp.index, dtype=float)
    cop[outdoor_temp >= 10] = 3.5
    cop[(outdoor_temp >= 0) & (outdoor_temp < 10)] = 2.5 + (outdoor_temp[(outdoor_temp >= 0) & (outdoor_temp < 10)] / 10) * (3.5 - 2.5)
    cop[outdoor_temp < 0] = 2.0 + (outdoor_temp[outdoor_temp < 0] / 10) * (2.5 - 2.0)
    
    electricity_demand = net_heat_demand_kwh / cop
    
    # If the air temperature is above 16°C, we don't need heating
    electricity_demand[outdoor_temp > 16] = 0

    df['heat'] = electricity_demand
    df['cop'] = cop
    
    return df

def estimate_hot_water_demand(df, daily_hot_water_demand_kwh=DAILY_HOT_WATER_USAGE_KWH):
    """
    Estimate hot water demand assuming that heating takes place in the hour of each day with the most solar generation. This is also likely to be a warmer hour when the COP of the heat pump is higher.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing time series data with a 'solar_generation' column (in kW or W).
    daily_hot_water_demand_kwh (float): Total hot water demand per day in kWh (default is 3 kWh).
    
    Returns:
    pd.DataFrame: DataFrame with a new 'hot_water_demand_kwh' column indicating the estimated hot water demand.
    """
    
    # Extract the hour and day from the index
    df['hour'] = df.index.hour
    df['day'] = df.index.date

    # Find the hour with the highest solar generation for each day
    max_solar_generation_per_day = df.groupby('day')['solar'].idxmax()

    # Create a new column for hot water demand, initialized to zero
    df['hot_water'] = 0

    # Assign the total daily hot water demand to the hour with the most solar generation
    df.loc[max_solar_generation_per_day, 'hot_water'] = daily_hot_water_demand_kwh

    # Optionally, drop the temporary 'day' and 'hour' columns if no longer needed
    df.drop(columns=['day', 'hour'], inplace=True)
    
    return df

def estimate_other_demand(df, daily_load_profile=DAILY_LOAD_PROFILE):
    ''' estimate the other electricity demand - which is basically a function of the hour of the day '''
    df['other_demand'] = daily_load_profile[df.index.hour] + VEHICLE_HOURLY_CONSUMPTION
    return df

def total_demand(df):
    df['net_demand'] = df['heat'] + df['hot_water'] + df['other_demand'] - df['solar']
    return df

def rolling_df(df, rolling_window=72):
    df = df.rolling(rolling_window).mean()
    df = df.dropna()
    return df

def plot_xy(df):
    ''' plot solar x and net_demand y'''
    df.plot(x='solar', y='net_demand', kind='scatter')
    plt.show()


def calculate_air_and_solar_energy_contribution(df):
    """
    Calculate the contribution of solar and air energy to total energy usage.
    
    This includes energy that is extracted from thin air by the heat pump 
    and the energy provided by solar panels. Plots both the old and new 
    self-sufficiency values.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing time series data with columns 
                       'solar', 'heat', and 'cop' (coefficient of performance).
    
    Returns:
    pd.DataFrame: Updated DataFrame with columns for the energy contribution 
                  from solar and air, and a graph showing self-sufficiency.
    """
    # Calculate the energy from thin air (based on heat demand and COP)
    df['electricity_for_heat'] = df['heat'] / df['cop']
    df['energy_from_air'] = df['heat'] - df['electricity_for_heat']
    
    # Ensure no negative values
    df['energy_from_air'][df['energy_from_air'] < 0] = 0

    # Combine solar energy and energy from air
    df['total_contribution'] = df['solar'] + df['energy_from_air']

    # Calculate daily self-sufficiency based on combined solar and air contribution
    df['day_of_year'] = df.index.dayofyear
    daily_contribution = df.groupby('day_of_year')['total_contribution'].sum()
    daily_net_demand = df.groupby('day_of_year')['net_demand'].sum()

    # New self-sufficiency (solar + air energy)
    daily_self_sufficiency = (daily_contribution / daily_net_demand) * 100
    daily_self_sufficiency = daily_self_sufficiency.clip(upper=100)

    # Calculate original self-sufficiency (solar-only, no air contribution)
    daily_solar_only = df.groupby('day_of_year')['solar'].sum()
    daily_self_sufficiency_solar_only = (daily_solar_only / daily_net_demand) * 100
    daily_self_sufficiency_solar_only = daily_self_sufficiency_solar_only.clip(upper=100)

    # Plot both self-sufficiency curves
    plt.figure()
    plt.plot(daily_self_sufficiency.index, daily_self_sufficiency, label='With Air Contribution', color='blue')
    plt.plot(daily_self_sufficiency_solar_only.index, daily_self_sufficiency_solar_only, linestyle='dotted', label='Solar Only', color='red')
    
    plt.xlabel('Day of the year')
    plt.ylabel('Self-sufficiency (%)')
    plt.title('Self-sufficiency with Solar and Air Energy Contribution')

    # Add key vertical lines for important months
    plt.axvline(x=90, color='green', linestyle='--', label='April 1')
    plt.axvline(x=121, color='yellow', linestyle='--', label='May 1')
    plt.axvline(x=212, color='orange', linestyle='--', label='August 1')
    plt.axvline(x=243, color='blue', linestyle='--', label='Sep 1')

    # Add horizontal line showing average self-sufficiency (with air)
    average_self_sufficiency = daily_self_sufficiency.mean()
    plt.axhline(y=average_self_sufficiency, color='grey', linestyle='--', label='Average self-sufficiency')

    # Set ylim to 0 to 100
    plt.ylim(0, 100)
    
    plt.legend()
    
    # Save the new plot as a PNG
    output_fp = 'self_sufficiency_air_and_solar_with_old.png'
    plt.savefig(output_fp)
    
    # calculate the mean self-sufficiency and print to console
    print(f"Mean self-sufficiency with air contribution: {daily_self_sufficiency.mean():.2f}%")
    print(f"Mean self-sufficiency with solar only: {daily_self_sufficiency_solar_only.mean():.2f}%")

    return df

def calculate_self_sufficiency(df):
    """
    Calculate daily self-sufficiency percentage based on solar generation and total demand.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing solar generation and total demand.
    
    Returns:
    pd.DataFrame: DataFrame with daily self-sufficiency values.
    """
    # Group by day
    df['day_of_year'] = df.index.dayofyear

    # Calculate daily solar generation and net demand
    daily_solar = df.groupby('day_of_year')['solar'].sum()
    daily_net_demand = df.groupby('day_of_year')['net_demand'].sum()

    # Calculate self-sufficiency as a percentage
    daily_self_sufficiency = (daily_solar / daily_net_demand) * 100
    daily_self_sufficiency = daily_self_sufficiency.clip(upper=100)  # Cap at 100%
    
    daily_self_sufficiency.plot()
    #add axis labels
    plt.xlabel('Day of the year')
    plt.ylabel('Self-sufficiency (%)')
    # can we add few vertical lines to show the start of key months
    plt.axvline(x=100, color='green', linestyle='--', label='April 10')
    plt.axvline(x=151, color='yellow', linestyle='--', label='May 31')
    plt.axvline(x=222, color='orange', linestyle='--', label='August 10')
    plt.axvline(x=258, color='blue', linestyle='--', label='Sep 15')
    
    # can we add a horizontal line to show the average self-sufficiency
    average_self_sufficiency = daily_self_sufficiency.mean()
    plt.axhline(y=average_self_sufficiency, color='grey', linestyle='--', label='Average self-sufficiency')
    
    plt.legend()
    # plt.show()
    fp = 'self_sufficiency_plot.png'
    # write to png
    plt.savefig(fp)
    
def calculate_daily_correlations(df):
    ''' evaluate the correlation between solar generation and gross demand '''
    df['gross_demand'] = df['net_demand'] + df['solar']
    # remove net_demand as it is not needed
    df.drop(columns=['net_demand'], inplace=True)
    # Group by day 
    df = df.resample('D').sum()
    # Add a month column to the DataFrame
    df['month'] = df.index.month

    # Initialize a dictionary to store correlations for each month
    monthly_correlations = {}

    # Month names for labeling
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Calculate the correlation between solar generation and gross demand for each month
    for month in range(1, 13):
        monthly_df = df[df['month'] == month]
        correlation = monthly_df['solar'].corr(monthly_df['gross_demand'])
        monthly_correlations[month_names[month - 1]] = correlation
        
    # Plot the monthly correlations
    monthly_correlations = pd.Series(monthly_correlations)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 7]})

    monthly_correlations.plot(kind='bar', ax=axes[0])
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Correlation between Solar Generation and Gross Demand')

    # Create a scatter plot for each month with each day as an observation
    # Add a month column to the DataFrame
    df['month'] = df.index.month

    # Month names for labeling
    df['month_name'] = df['month'].apply(lambda x: month_names[x - 1])

    # Create a FacetGrid for the scatter plots
    g = sns.FacetGrid(df, col='month_name', col_wrap=4, height=4, aspect=1.5)
    g.map(sns.scatterplot, 'solar', 'gross_demand', alpha=0.5)

    # Add a line of best fit to each scatter plot
    g.map_dataframe(sns.regplot, x='solar', y='gross_demand', scatter=False, truncate=False, color='red')

    # Set axis labels and titles
    g.set_titles('{col_name}')
    g.set_xlabels('Solar Generation (kWh)')
    g.set_ylabels('Gross Demand (kWh)')

    # Save the FacetGrid plot to a temporary file
    facet_grid_fp = 'monthly_corr_scatter_plot_facets.png'
    g.savefig(facet_grid_fp)
    plt.close()

    # Load the FacetGrid plot and add it to the second subplot
    facet_grid_img = plt.imread(facet_grid_fp)
    axes[1].imshow(facet_grid_img)
    axes[1].axis('off')  # Hide the axes for the image

    # Adjust the aspect ratio of the right-hand plot to be full height
    axes[1].set_aspect(aspect='auto')

    # Save the combined plot
    combined_fp = 'combined_plot.png'
    plt.savefig(combined_fp)
    plt.close()


def calculate_solar_heatpump_power(df: pd.DataFrame):
    """
    Identify in which calendar months solar makes a significant contribution to heat demand.
    The solar contribution is calculated as the proportion of solar power used directly to meet heat demand.
    """
    # Calculate total heat demand (heat + hot water)
    df['total_heat_demand'] = df['heat'] + df['hot_water']

    # Determine the solar contribution to hot water first
    df['solar_contribution_hot_water'] = df[['solar', 'hot_water']].min(axis=1)

    # Calculate the remaining solar power after meeting hot water demand
    df['remaining_solar'] = df['solar'] - df['solar_contribution_hot_water']

    # Determine the solar contribution to heating from the remaining solar power
    df['solar_contribution_heat'] = df[['remaining_solar', 'heat']].min(axis=1)
    
    # Drop the temporary 'remaining_solar' column
    df.drop(columns=['remaining_solar'], inplace=True)
    df['month'] = df.index.month
    monthly_df = df[['heat', 'hot_water', 'solar_contribution_heat', 'solar_contribution_hot_water', 'month']].groupby('month').mean()
    
    monthly_df['solar_fraction_heat'] = monthly_df['solar_contribution_heat'] / monthly_df['heat'] * 100
    # set solar_fraction_heat to 0 for June to Sept - when minimal heat is needed
    monthly_df.loc[6:9, 'solar_fraction_heat'] = 0
    
    monthly_df['solar_fraction_hot_water'] = monthly_df['solar_contribution_hot_water'] / monthly_df['hot_water'] * 100
    
    import pdb; pdb.set_trace()
   
    # Plot the solar contribution to heat demand
    plt.figure(figsize=(12, 6))
    monthly_df[['solar_fraction_hot_water', 'solar_fraction_heat']].plot(kind='bar', color=['orange', 'skyblue'], title='Monthly Solar Contribution to Hot Water and Heat Demand')
    plt.ylabel('Solar Contribution (%)')
    plt.xlabel('Month')
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.legend(['Hot Water', 'Space Heating'])
    plt.tight_layout()
    
    # write to png
    fp = 'monthly_solar_contribution.png'
    plt.savefig(fp)
    
    return monthly_df

def clean_up(df):
    to_drop = ['ghi', 'dhi', 'dni', 'hot_water', 'other_demand', 'wind_speed', 'heat', 'temperature']
    df.drop(columns=to_drop, inplace=True)
    return df

def calculate_wind_air_and_solar_energy_contribution(df, wind_turbine_capacity_kw=1.2, wind_efficiency=0.3, cut_in_speed=3, rated_speed=12, cut_out_speed=25):
    """
    Calculate the contribution of wind, solar, and air energy to total energy usage.
    
    This includes energy that is extracted from thin air by the heat pump, 
    energy provided by solar panels, and the energy contribution from a wind turbine.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing time series data with columns 
                       'solar', 'heat', 'cop' (coefficient of performance), and 'wind_speed'.
    wind_turbine_capacity_kw (float): Maximum capacity of the wind turbine in kW (default is 1.2 kW for Rutland 1200).
    wind_efficiency (float): Efficiency factor of the wind turbine (default is 0.3).
    
    Returns:
    pd.DataFrame: Updated DataFrame with columns for the energy contribution 
                  from wind, solar, and air, and a graph showing self-sufficiency.
    """
    # Calculate the energy from thin air (based on heat demand and COP)
    df['electricity_for_heat'] = df['heat'] / df['cop']
    df['energy_from_air'] = df['heat'] - df['electricity_for_heat']
    
    # Ensure no negative values for energy from air
    df['energy_from_air'][df['energy_from_air'] < 0] = 0

    
    df['wind_speed_m_s'] = df['wind_speed'] / 3.6
    
    # Wind turbine power curve: cut-in, rated, and cut-out speeds
    wind_speed_m_s = df['wind_speed_m_s']
    
    # Calculate wind power using a realistic turbine power curve
    df['wind_power'] = 0  # Initialize with zero
    
    # Wind turbine produces power between cut-in and cut-out speeds
    df.loc[(wind_speed_m_s >= cut_in_speed) & (wind_speed_m_s < rated_speed), 'wind_power'] = (
        ((wind_speed_m_s[(wind_speed_m_s >= cut_in_speed) & (wind_speed_m_s < rated_speed)] - cut_in_speed) / (rated_speed - cut_in_speed)) * wind_turbine_capacity_kw
    )
    
    # Wind turbine produces rated power at or above rated speed but below cut-out speed
    df.loc[(wind_speed_m_s >= rated_speed) & (wind_speed_m_s <= cut_out_speed), 'wind_power'] = wind_turbine_capacity_kw
    
    # Ensure the turbine stops producing power above the cut-out speed
    df.loc[wind_speed_m_s > cut_out_speed, 'wind_power'] = 0

    

    # Combine wind, solar, and energy from air
    df['total_contribution'] = df['solar'] + df['wind_power'] + df['energy_from_air']
    

    # Calculate daily self-sufficiency based on combined wind, solar, and air contribution
    df['day_of_year'] = df.index.dayofyear
    daily_contribution = df.groupby('day_of_year')['total_contribution'].sum()
    daily_net_demand = df.groupby('day_of_year')['net_demand'].sum()

    # New self-sufficiency (wind + solar + air energy)
    daily_self_sufficiency = (daily_contribution / daily_net_demand) * 100
    daily_self_sufficiency = daily_self_sufficiency.clip(upper=100)

    # Calculate original self-sufficiency (solar-only, no wind or air contribution)
    daily_solar_only = df.groupby('day_of_year')['solar'].sum()
    daily_self_sufficiency_solar_only = (daily_solar_only / daily_net_demand) * 100
    daily_self_sufficiency_solar_only = daily_self_sufficiency_solar_only.clip(upper=100)

    # Plot both self-sufficiency curves
    plt.figure()
    plt.plot(daily_self_sufficiency.index, daily_self_sufficiency, label='With Wind and Air Contribution', color='blue')
    plt.plot(daily_self_sufficiency_solar_only.index, daily_self_sufficiency_solar_only, linestyle='dotted', label='Solar Only', color='red')
    
    plt.xlabel('Day of the year')
    plt.ylabel('Self-sufficiency (%)')
    plt.title('Self-sufficiency with Wind/Solar Hybrid')

    # Add key vertical lines for important months
    plt.axvline(x=90, color='green', linestyle='--', label='April 1')
    plt.axvline(x=121, color='yellow', linestyle='--', label='May 1')
    plt.axvline(x=212, color='orange', linestyle='--', label='August 1')
    plt.axvline(x=243, color='blue', linestyle='--', label='Sep 1')

    # Add horizontal line showing average self-sufficiency (with wind and air)
    average_self_sufficiency = daily_self_sufficiency.mean()
    plt.axhline(y=average_self_sufficiency, color='grey', linestyle='--', label='Average self-sufficiency')

    # Set ylim to 0 to 100
    plt.ylim(0, 100)
    
    plt.legend()
    
    # Save the new plot as a PNG
    output_fp = 'self_sufficiency_wind_air_and_solar.png'
    plt.savefig(output_fp)
    
    # Calculate and print mean self-sufficiency values
    print(f"Mean self-sufficiency with wind and air contribution: {daily_self_sufficiency.mean():.2f}%")
    print(f"Mean self-sufficiency with solar only: {daily_self_sufficiency_solar_only.mean():.2f}%")

    return df


if __name__ == "__main__":
    df = open_data()
    df = estimate_solar_power(df)
    df = estimate_heat_demand(df)
    df = estimate_hot_water_demand(df)
    df = estimate_other_demand(df)
    df = total_demand(df)
    
    # calculate_solar_heatpump_power(df.copy())
    # calculate_daily_correlations(df.copy())
    
    # calculate_air_and_solar_energy_contribution(df.copy())
    
    calculate_wind_air_and_solar_energy_contribution(df.copy())
