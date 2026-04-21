import yfinance as yf
voo = yf.Ticker("VOO") # ETF that tracks S&P 500

current_price = voo.info.get('regularMarketPrice')
print(f"Current VOO Price: {current_price}")
live_data = voo.history(period="1d", interval="1m")
print(live_data.tail()) # bottom row is most current data
