import pandas as pd

def process_weather_data(input_file, output_file):
    # Load the JSON data into a DataFrame
    df = pd.read_json(input_file)

    # Convert the 'ts' (timestamp) to a datetime index
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Resample to weekly data, taking the mean of each week
    # FIXME: here we take the mean, which is correct for wind and temperature, but maybe for incident solar we need a sum?
    weekly_df = df.resample('W').mean()

    # Reset the index to turn the index into a column again
    weekly_df.reset_index(inplace=True)

    # Rename columns for clarity if needed
    weekly_df.rename(columns={'t': 'temperature', 'w': 'wind_speed', 's': 'ghi', 'd': 'dhi', 'n': 'dni'}, inplace=True)

    # Save the processed DataFrame to a new JSON file
    weekly_df.to_json(output_file, orient='records', date_format='iso')

if __name__ == "__main__":
    process_weather_data('weather_history.json', 'weekly_weather_history.json')

