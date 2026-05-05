import yfinance as yf
# voo = yf.Ticker("VOO") # ETF that tracks S&P 500

# current_price = voo.info.get('regularMarketPrice')
# print(f"Current VOO Price: {current_price}")
# live_data = voo.history(period="1d", interval="1m")
# print(live_data.tail()) # bottom row is most current data

# Get Stock data for S&P 500, Nasdaq, Dow Jones, Crude Oil, VIX
data = yf.download("^GSPC ^IXIC ^DJI CL=F ^VIX", start="2000-01-01", end="2026-05-05", interval="1d", group_by="Date")

# Fix weird column stacking problem
df = data.stack(level=0).reset_index()
df = df.rename(columns={'level_1': 'Ticker'})

# Make names more readable and save
df.loc[df['Ticker'] == "^GSPC", 'Ticker'] = "SandP500"
df.loc[df['Ticker'] == "^IXIC", 'Ticker'] = "Nasdaq"
df.loc[df['Ticker'] == "^DJI", 'Ticker'] = "DowJones"
df.loc[df['Ticker'] == "CL=F", 'Ticker'] = "Oil"
df.loc[df['Ticker'] == "^VIX", 'Ticker'] = "VIX"
df = df.rename(columns={'Ticker': 'Name'})
df.set_index("Date", inplace=True)
df.to_csv("..\\data\\stock_data.csv")