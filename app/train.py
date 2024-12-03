import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import optimizers, losses, nn
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully!")
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None

    # Check for missing values and handle them
    if data.isnull().sum().any():
        print("Missing values found! Removing or filling them...")
        data.fillna(method='ffill', inplace=True)
        data.dropna(inplace=True)

    # Normalize the relevant columns
    scaler = MinMaxScaler()
    data[['Close', 'avg_sentiment']] = scaler.fit_transform(data[['Close', 'avg_sentiment']])

    # Prepare sequences for LSTM
    sequence_length = 50
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[['Close', 'avg_sentiment']].iloc[i-sequence_length:i].values)
        y.append(data['Close'].iloc[i])

    X, y = np.array(X), np.array(y)

    # Ensure no NaNs in X or y
    assert not np.isnan(X).any(), "Feature data (X) contains NaN values!"
    assert not np.isnan(y).any(), "Target data (y) contains NaN values!"

    return X, y

# Function to build the LSTM model
def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation=nn.tanh, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50, activation=nn.tanh),
        tf.keras.layers.Dense(1)  # Output layer for predicting the stock price
    ])

    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredError())

    return model

# Function to train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=30, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history

# Function to save the trained model
def save_model(model, model_path):
    model.save(model_path)  # Save the model as an H5 file
    print(f"Model saved successfully at {model_path}")

# Function to evaluate the model and compute MSE
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    assert not np.isnan(train_predictions).any(), "Train predictions contain NaN values!"
    assert not np.isnan(test_predictions).any(), "Test predictions contain NaN values!"

    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

# Main function to execute the entire process
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    data_file_path = os.path.join(script_dir, "../data/merged_stock_sentiment_data.csv")

    # Load and preprocess data
    X, y = load_and_preprocess_data(data_file_path)
    if X is None or y is None:
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train the model
    train_model(model, X_train, y_train, X_test, y_test)

    # Save the trained model dynamically
    model_save_dir = os.path.join(script_dir, "../model")
    if not os.path.exists(model_save_dir):  # Ensure the directory exists
        os.makedirs(model_save_dir)

    model_save_path = os.path.join(model_save_dir, 'stock_sentiment_model.h5')
    save_model(model, model_save_path)


    # Evaluate the model
    evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
