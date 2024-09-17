import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

VEHICLE_ANNUAL_MILEAGE = 10000  # miles
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
    

def clean_up(df):
    to_drop = ['ghi', 'dhi', 'dni', 'hot_water', 'other_demand', 'wind_speed', 'heat', 'temperature']
    df.drop(columns=to_drop, inplace=True)
    return df

if __name__ == "__main__":
    df = open_data()
    df = estimate_solar_power(df)
    df = estimate_heat_demand(df)
    df = estimate_hot_water_demand(df)
    df = estimate_other_demand(df)
    df = total_demand(df)
    df = clean_up(df)
    calculate_self_sufficiency(df)

    df = rolling_df(df)
    # plot_xy(df)
    print(df)
