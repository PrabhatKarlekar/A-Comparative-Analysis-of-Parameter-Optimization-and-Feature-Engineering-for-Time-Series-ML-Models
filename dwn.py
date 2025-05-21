import yfinance as yf

# Define the stock ticker and date range
ticker_symbol = 'RELIANCE.NS'  # Example: Apple Inc.
start_date = '1996-02-02'
end_date = '2024-12-31'

# Download the stock data
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Save to CSV
data.to_csv(f"{ticker_symbol}_stock_data.csv")

print(f"Stock data for {ticker_symbol} saved to {ticker_symbol}_stock_data.csv")