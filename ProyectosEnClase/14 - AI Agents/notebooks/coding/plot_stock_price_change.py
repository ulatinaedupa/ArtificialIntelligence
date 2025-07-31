# filename: plot_stock_price_change.py
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stock ticker and the time period
ticker = "AAPL"  # Apple Inc.
start_date = "2023-01-01"
end_date = "2023-10-15"

# Download the stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Check if the data is available
if stock_data.empty:
    print(f"Data for {ticker} from {start_date} to {end_date} is not available")
else:
    # Plot the closing prices
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label=f"{ticker} Stock Price")
    plt.title(f"{ticker} Stock Price Change from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig("stock_price_change.png")
    plt.show()