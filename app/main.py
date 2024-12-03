import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from test import predict_stock_price  # Import your prediction function

# Function to create the dropdown with US stock tickers
def get_us_stock_tickers():
    # Predefined list of popular US stock tickers
    # You can add more tickers to this list based on your requirements
    tickers = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'SPY', 'AMT', 
        'AMD', 'INTC', 'WMT', 'DIS', 'BA', 'PYPL', 'BABA', 'V', 'JNJ', 'KO', 'PFE', 
        'MS', 'VZ', 'C', 'GS', 'IBM', 'GM', 'GE', 'CAT', 'CSCO', 'MCD', 'LMT', 'RTX', 
        'UPS', 'UNH', 'HD', 'MA', 'CVX', 'XOM', 'SQ', 'SPG', 'AT&T', 'SO', 'T', 'ORCL'
    ]
    return tickers

def create_streamlit_app():
    st.title("Stock Prediction App")

    # Get all US stock tickers
    us_stock_tickers = get_us_stock_tickers()

    # Dropdown for selecting stock symbol from the list of US tickers
    stock_symbol = st.selectbox("Select the stock symbol", options=us_stock_tickers)

    # Fixed time period: 1 month
    time_period = "1mo"

    # Submit button to trigger prediction
    submit_button = st.button("Submit")

    if submit_button:
        # Call the prediction function and get the predicted stock price
        predicted_price = predict_stock_price(stock_symbol)

        if predicted_price is not None:
            # Fetch historical stock data for the last month (1mo)
            stock_data = yf.download(stock_symbol, period=time_period, interval='1d')  # Fetch data for the past month
            stock_data = stock_data[['Close']]  # Use only the closing price

            # Display the historical stock data using Streamlit's built-in line chart
            st.subheader(f"Stock Price Data for {stock_symbol} ({time_period})")
            st.line_chart(stock_data['Close'])

            # Show the predicted price
            st.subheader(f"Predicted stock price for the next day: ${predicted_price:.2f}")

            # Create a plot using matplotlib for both historical and predicted data
            fig, ax = plt.subplots(figsize=(10, 5))

            # Plot historical data
            ax.plot(stock_data.index, stock_data['Close'], label='Historical Stock Price', color='blue', linewidth=2)

            # Add predicted stock price as a new point (next day's prediction)
            prediction_date = stock_data.index[-1] + pd.Timedelta(days=1)  # Calculate the next date
            ax.scatter(prediction_date, predicted_price, color='red', label='Predicted Stock Price', zorder=5)

            # Customizing the plot
            ax.set_title(f"{stock_symbol} Stock Price (Historical and Predicted)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()

            # Show the plot using Streamlit
            st.pyplot(fig)

        else:
            st.write("Error: Could not predict the stock price.")

if __name__ == "__main__":
    # Set the layout and page title
    st.set_page_config(layout="wide", page_title="Stock Prediction App")

    # Run the app function to create the UI
    create_streamlit_app()
