from fredapi import Fred
fred = Fred(api_key='your_api')
sp500 = fred.get_series('SP500')
cpi_month = fred.get_series('CPIAUCSL')