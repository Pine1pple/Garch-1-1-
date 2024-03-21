import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model

file_path = 'C:/Users/maait/Downloads/BTC-USD.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

def perform_analysis(df):

    # Display the first few rows of the DataFrame
    print("First few rows of the DataFrame:")
    print(df.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())

    # Line plot of closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], color='blue')
    plt.title('Bitcoin Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()

    # Histogram of log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    plt.figure(figsize=(8, 6))
    plt.hist(log_returns, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Log Returns')
    plt.xlabel('Log Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Autocorrelation plot of log returns
    plt.figure(figsize=(8, 6))
    pd.plotting.autocorrelation_plot(log_returns)
    plt.title('Autocorrelation of Log Returns')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.show()

# Perform analysis on the DataFrame
perform_analysis(df)

df.set_index('Date', inplace=True)

# Calculate log returns
log_returns = np.log(df['Close']).diff().dropna()

# Fit GARCH(1,1) model
garch_model = arch_model(log_returns, vol='Garch', p=1, q=1)
result = garch_model.fit()

# Display model summary
print(result.summary())

# Plot the standardized residuals
result.plot()

# Plot the volatility
fig, ax = plt.subplots()
ax.plot(result.conditional_volatility)
ax.set_title('Conditional Volatility (GARCH(1,1))')
plt.show()
