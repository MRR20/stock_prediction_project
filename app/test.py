import numpy as np
import pandas as pd
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import requests
from datetime import datetime
import tensorflow as tf
import os

# Download the VADER lexicon (only needed once)
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def predict_stock_price(stock_symbol: str, num_days: int = 7):
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    
    # Construct the model file path using the script directory
    model_file_path = os.path.join(script_dir, "../model/stock_sentiment_model.h5")  # Path to the model (note the .h5 extension)
    
    # Step 1: Get the latest stock data
    stock_data = yf.download(stock_symbol, period='1d', interval='1d')  # Get the most recent stock data
    stock_data = stock_data[['Close']]  # Use only the 'Close' price for simplicity

    # Step 2: Get the latest news sentiment data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - pd.Timedelta(days=num_days)).strftime('%Y-%m-%d')

    api_key = 'f3e342dc-477b-4784-bba2-a0916569947b'  # Replace with your actual API key
    base_url = 'https://content.guardianapis.com/search'

    # Fetch news articles
    news_data = []
    current_date = pd.to_datetime(start_date)

    while current_date <= pd.to_datetime(end_date):
        date_str = current_date.strftime('%Y-%m-%d')
        params = {
            'section': 'business',
            'page-size': 200,
            'from-date': date_str,
            'to-date': date_str,
            'show-fields': 'body',
            'api-key': api_key,
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json().get("response", {}).get("results", [])
            if not results:
                print(f"No results found for {date_str}")
            for article in results:
                # Check if the 'fields' and 'body' keys exist
                if "fields" in article and "body" in article["fields"]:
                    news_data.append({'date': date_str, 'content': article["fields"]["body"]})
        else:
            print(f"Failed to fetch data for {date_str}: {response.status_code}")

        current_date += pd.Timedelta(days=1)

    # Analyze sentiment
    if news_data:
        news_df = pd.DataFrame(news_data)
        news_df['sentiment'] = news_df['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        sentiment_data = news_df.groupby('date')['sentiment'].mean().reset_index()
    else:
        print("No news data found.")
        sentiment_data = pd.DataFrame(columns=['date', 'sentiment'])

    # Step 3: Prepare the data for prediction (latest stock price and sentiment)
    if not stock_data.empty and not sentiment_data.empty:
        latest_stock_price = stock_data['Close'].values[-1]  # Most recent stock price
        latest_sentiment = sentiment_data['sentiment'].values[-1]  # Most recent sentiment score

        # Ensure that the values are scalar and not arrays
        latest_stock_price = latest_stock_price.item()  # Convert to scalar
        latest_sentiment = latest_sentiment.item()  # Convert to scalar

        # Combine stock price and sentiment into one feature array
        X = np.array([[[latest_stock_price, latest_sentiment]]])  # Reshape to (1, 1, 2)

        # Step 4: Load the pre-trained model using Keras
        model = tf.keras.models.load_model(model_file_path)  # Load the .h5 model using tf.keras.models.load_model

        # Step 5: Predict the stock price for the next day using the loaded model
        prediction_input = tf.convert_to_tensor(X, dtype=tf.float32)
        predicted_price_normalized = model.predict(prediction_input)  # Make the prediction

        # Step 6: Denormalize the predicted price back to the actual price
        # Assuming you have the scaler used during training to normalize the data
        # In this case, we just use the stock price scaler to revert the prediction
        predicted_price_actual = predicted_price_normalized[0][0] * (stock_data['Close'].max() - stock_data['Close'].min()) + stock_data['Close'].min()

        # Return the predicted stock price for the next day
        return predicted_price_actual.item()  # Return the actual predicted value

    else:
        print("Error: Stock or sentiment data is missing!")
        return None

# Example usage:
# predicted_price = predict_stock_price('AAPL')
# print(f"Predicted stock price for the next day: ${predicted_price:.2f}")
