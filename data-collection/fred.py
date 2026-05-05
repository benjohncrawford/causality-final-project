from fredapi import Fred
import json
with open("secrets.json", "r") as f:
    data = json.load(f)
fred = Fred(api_key=data["fred"])
sp500 = fred.get_series('SP500')
cpi_month = fred.get_series('CPIAUCSL')