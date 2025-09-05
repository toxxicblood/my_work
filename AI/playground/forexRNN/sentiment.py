import pandas as pd
import requests
import io
import json

API_KEY = 'HERFSC5T3RR7T4SP9'

def get_sentiment_data():
    """
    Downloads historical news and sentiment data from the Alpha Vantage API.
    """
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={API_KEY}&time_from=20200101T0000&time_to=20250322T0000'
    r = requests.get(url)
    data = r.json()
    
    if 'feed' not in data:
        print("Error: 'feed' not in API response. The response was:")
        print(json.dumps(data, indent=4))
        return

    df = pd.DataFrame(data['feed'])
    
    # Resample to hourly frequency and forward-fill missing values
    df['time_published'] = pd.to_datetime(df['time_published'], format='mixed', utc=True)
    df = df.set_index('time_published')
    
    # Remove duplicate timestamps
    print(f"Number of duplicate timestamps before removal: {df.index.duplicated().sum()}")
    df = df[~df.index.duplicated(keep='first')]
    print(f"Number of duplicate timestamps after removal: {df.index.duplicated().sum()}")

    df = df.resample('H').ffill()
    
    df.to_csv('histdata/sentiment.csv')
    print("Sentiment data saved to histdata/sentiment.csv")

if __name__ == '__main__':
    get_sentiment_data()