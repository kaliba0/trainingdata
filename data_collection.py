import requests
import json
from datetime import datetime, timedelta

def fetch_historical_data(symbol, interval, start_date, end_date, output_file):
    """
    Fetches historical data for a given symbol and time interval from Binance API and saves it to a JSON file.

    Args:
        symbol (str): The symbol of the cryptocurrency (e.g., 'BTCUSDT').
        interval (str): The interval for the data (e.g., '1d' for daily data).
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        output_file (str): The path of the output JSON file to save the data.
    """
    # Convert dates to timestamps
    start_time = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_time = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000  # Maximum number of data points per request
    }

    all_data = []

    while start_time < end_time:
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        # Update the start_time to fetch the next batch of data
        start_time = data[-1][0] + 1
        params['startTime'] = start_time

    # Convert data to a readable format and save to JSON
    formatted_data = [
        {
            "timestamp": datetime.fromtimestamp(item[0] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5])
        }
        for item in all_data
    ]

    # Save the data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

    print(f"Data saved to {output_file}")


# Example usage
symbol = 'BTCEUR'
interval = '1d'
start_date = '2023-01-01'
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
output_file = 'historical_data.json' 

fetch_historical_data(symbol, interval, start_date, end_date, output_file)
