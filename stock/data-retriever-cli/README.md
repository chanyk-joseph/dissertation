### HK Market Stock Quote
A single executable for serving quote requests via RestAPI / updating HSI components quotes to mysql database       

## Download
[Windows-x64](https://raw.githubusercontent.com/chanyk-joseph/dissertation/master/stock/data-retriever-cli/data-retriever-cli_windows_x64.exe)<br/>
[Linux-x64](https://raw.githubusercontent.com/chanyk-joseph/dissertation/master/stock/data-retriever-cli/data-retriever-cli_linux_x64)<br/>
[Mac-x64](https://raw.githubusercontent.com/chanyk-joseph/dissertation/master/stock/data-retriever-cli/data-retriever-cli_mac_x64)
<br/>
# Stock Quote Rest API Server Usage
```
./data-retriever-cli web -p 8888
```
#### Endpoints
---
* [/quote/&lt;stock_symbol&gt;](#simple-quote)
* [/quote/&lt;stock_symbol&gt;?raw=true](#raw-quote)
* [/history/&lt;stock_symbol&gt;?starttime=&lt;unix_time&gt;&endtime=&lt;unix_time&gt;&resolution=&lt;resolution&gt;&provider=&lt;provider&gt;](#history)
* [/hsicomponents](#hsi-components)
* [/hsicomponents/quote](#hsi-components-quote)

**&lt;stock_symbol&gt;**: 700, 0700, 00700, 700.HK, 700:HK etc <br/>
**&lt;unix_time&gt;**: Unix Epoch Time In Seconds, eg: 01 Jun 2018 00:00:00 => 1527811200 <br/>
**&lt;resolution&gt;**: day / 4hours / hour / 30minutes / 15minutes / 5minutes / minute (p.s. default = day) <br/>
**&lt;provider&gt;**: yahoo / &lt;local-folder-path&gt; (p.s. default = yahoo) <br/>
**&lt;local-folder-path&gt;**: The local parent folder which stores the CSV files. (p.s. Use to serve data from local CSV files)

#### Rest API Sample Responses
---
###### <a name="simple-quote"></a>http://127.0.0.1:8888/quote/700
```json
{
	"symbol": "00700.HK",
	"quotes": {
		"aastocks": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20170000
		},
		"bloomberg": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20167800
		},
		"hkex": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20170000
		},
		"investtab": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20167800
		}
	}
}
```
---
###### <a name="raw-quote"></a>http://127.0.0.1:8888/quote/700?raw=true
```json
{
	"symbol": "00700.HK",
	"quotes": {
		"aastocks": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20170000
		},
		"bloomberg": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20167800
		},
		"hkex": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20170000
		},
		"investtab": {
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20167800
		}
	},
	"raw": {
		"aastocks": {
			"symbol": "00700.HK",
			"last_traded_price": 397.4,
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"bid": 397.4,
			"ask": 397.6,
			"privious_close": 396.8,
			"volume": "20.17M",
			"lot_size": 100,
			"turnover": "7.99B",
			"pe": 43.63,
			"yield": "0.22%",
			"dividend_payout": "9.661%",
			"eps": 9.109,
			"market_capital": "3,776.93B",
			"net_asset_value": 32.32,
			"low_52_weeks": 260.4,
			"high_52_weeks": 476.6
		},
		"bloomberg": {
			"symbol": "700:HK",
			"company_name": "Tencent Holdings Ltd",
			"exchange": "Hong Kong",
			"market_cap": "3.777T",
			"currency": "HKD",
			"previous_close": 396.8,
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"last_traded_price": 397.4,
			"volume": 20167800,
			"low_52_weeks": 260.4,
			"high_52_weeks": 476.6,
			"PE": 38.64,
			"best_PE": 34.8384,
			"best_PEG": 1.5205,
			"shares_outstanding": "9.5B",
			"price_to_book_ratio": 11.2832,
			"price_to_sales_ratio": 11.8588,
			"one_year_return": "42.13%",
			"average_volume_30_days": 24376530,
			"EPS": 8.53,
			"best_EPS_in_current_year": 8.89,
			"dividend": "0.22%",
			"last_dividend_reported": 0.88
		},
		"hkex": {
			"updatetime": "2018年6月22日16:08",
			"nm_s": "騰訊控股",
			"nm": "騰訊控股有限公司",
			"sym": "700",
			"ric": "0700.HK",
			"eps": 7.5986,
			"eps_ccy": "RMB",
			"pe": "42.23",
			"div_yield": "0.22",
			"mkt_cap": "3,776.92",
			"mkt_cap_u": "B",
			"ls": "397.400",
			"hc": "396.800",
			"op": "395.600",
			"hi": "400.400",
			"lo": "391.200",
			"hi52": "476.600",
			"lo52": "260.379",
			"am": "7.99",
			"am_u": "B",
			"vo": "20.17",
			"vo_u": "M",
			"bd": "397.400",
			"as": "397.600"
		},
		"investtab": {
			"symbol": "00700:HK",
			"as_of_date": "2018-06-22T00:00:00",
			"open": 395.6,
			"low": 391.2,
			"high": 400.4,
			"close": 397.4,
			"volume": 20167800,
			"volume_avg_20d": 21910617.5,
			"low_10d": 391.2,
			"low_250d": 260.4,
			"high_10d": 426.6,
			"high_250d": 476.6,
			"sma10": 407.98,
			"sma20": 408.92,
			"sma50": 403.108,
			"sma100": 421.658,
			"sma250": 380.5824,
			"bbands_lower": 388.9647,
			"bbands_upper": 428.8753,
			"di_minus": 32.6945,
			"di_plus": 19.0103,
			"macd": -1.3004,
			"macd_signal": 1.1284,
			"macd_hist": -2.4288,
			"stc_k": 11.1595,
			"stc_d": 12.9074,
			"adx": 15.6447,
			"rsi": 42.1564,
			"prices_gap": 0,
			"change_from_open": 1.8,
			"change_from_open_pct": 0.455,
			"change_from_prev_close": 0.6,
			"change_from_prev_close_pct": 0.1512
		}
	}
}
```

---
###### <a name="history"></a>http://127.0.0.1:8888/history/700?resolution=day&starttime=946684800&endtime=1529971200
```json
[{
		"open": 0.875,
		"high": 0.925,
		"low": 0.815,
		"close": 0.83,
		"adjusted_close": 0.749649,
		"volume": 2198875000,
		"date": "2004-06-16T00:00:00.000+0000"
	}, {
		"open": 0.83,
		"high": 0.875,
		"low": 0.825,
		"close": 0.845,
		"adjusted_close": 0.763197,
		"volume": 419007500,
		"date": "2004-06-17T00:00:00.000+0000"
	}, ... {
		"open": 395.600006,
		"high": 400.399994,
		"low": 391.200012,
		"close": 397.399994,
		"adjusted_close": 397.399994,
		"volume": 20167800,
		"date": "2018-06-22T00:00:00.000+0000"
	}
]
```

---
###### <a name="use-local-csv-history"></a>http://127.0.0.1:8888/history/700?starttime=946684800&endtime=1529971200&provider=./local-sample-csvs
Assumptions:
* The symbol is the sub-folder name under the local provider folder
* Example Folder: https://github.com/chanyk-joseph/dissertation/tree/master/stock/data-retriever-cli/local-sample-csvs
* The csv file name must in one of this format: ["YYYY-MM-ddTHH:mm:ss", "YYYY-MM-dd HH:mm:ss", "dd-MM-YYYY HH:mm:ss", "YYYY/MM/dd", "YYYY-MM-dd", "YYYY.MM.dd", "dd/MM/YYYY", "dd-MM-YYYY", "dd.MM.YYYY"]
* The first row of each CSV is the header
* The first column of each CSV is the time which must be in one of this format: ["YYYY-MM-ddTHH:mm:ss", "YYYY-MM-dd HH:mm:ss", "dd-MM-YYYY HH:mm:ss", "YYYY/MM/dd", "YYYY-MM-dd", "YYYY.MM.dd", "dd/MM/YYYY", "dd-MM-YYYY", "dd.MM.YYYY"]
```json
[{
		"Amount": "0.00",
		"BP1": "20857",
		"BV1": "2",
		"High": "21019",
		"LastClose": "20854",
		"Low": "20820",
		"Open": "20820",
		"OpenInt": "6810",
		"Price": "20862",
		"Price2": "0.00000",
		"Price3": "20855",
		"SP1": "20861",
		"SV1": "3",
		"Time": "2017-05-20 09:15:11",
		"TotalAmount": "0.000",
		"TotalVol": "8328",
		"Volume": "11",
		"isBuy": "1"
	}, {
		"Amount": "0.00",
		"BP1": "20857",
		"BV1": "2",
		"High": "21019",
		"LastClose": "20854",
		"Low": "20820",
		"Open": "20820",
		"OpenInt": "6810",
		"Price": "20857",
		"Price2": "0.00000",
		"Price3": "20855",
		"SP1": "20861",
		"SV1": "3",
		"Time": "2017-05-20 09:15:11",
		"TotalAmount": "0.000",
		"TotalVol": "8338",
		"Volume": "10",
		"isBuy": "0"
	}, ... {
		"Amount": "0.00",
		"BP1": "20857",
		"BV1": "2",
		"High": "21019",
		"LastClose": "20854",
		"Low": "20820",
		"Open": "20820",
		"OpenInt": "6810",
		"Price": "20855",
		"Price2": "0.00000",
		"Price3": "20855",
		"SP1": "20861",
		"SV1": "3",
		"Time": "2017-05-20 09:15:12",
		"TotalAmount": "0.000",
		"TotalVol": "8352",
		"Volume": "14",
		"isBuy": "0"
	}
]
```

---
###### <a name="hsi-components"></a>http://127.0.0.1:8888/hsicomponents
```json
["00941.HK","01038.HK","01044.HK","01088.HK","01093.HK","01109.HK","01113.HK","01299.HK","01398.HK","01928.HK","01997.HK","02007.HK","02018.HK","02318.HK","02319.HK","02382.HK","02388.HK","02628.HK","03328.HK","03988.HK","00001.HK","00002.HK","00003.HK","00005.HK","00006.HK","00011.HK","00012.HK","00016.HK","00017.HK","00019.HK","00023.HK","00027.HK","00066.HK","00083.HK","00101.HK","00144.HK","00151.HK","00175.HK","00267.HK","00288.HK","00386.HK","00388.HK","00688.HK","00700.HK","00762.HK","00823.HK","00836.HK","00857.HK","00883.HK","00939.HK"]
```

---
###### <a name="hsi-components-quote"></a>http://127.0.0.1:8888/hsicomponents/quote
```json
{
	"quotes": [{
			"symbol": "00941.HK",
			"quotes": {
				"aastocks": {
					"open": 70,
					"low": 68.3,
					"high": 70,
					"close": 68.7,
					"volume": 15410000
				}, 
				"hkex": {
					"open": 70,
					"low": 68.3,
					"high": 70,
					"close": 68.7,
					"volume": 15410000
				}, ...
			}
		}, {
			"symbol": "01038.HK",
			"quotes": {
				"aastocks": {
					"open": 57.9,
					"low": 57,
					"high": 57.95,
					"close": 57.45,
					"volume": 2610000
				}, 
				"investtab": {
					"open": 57.9,
					"low": 57,
					"high": 57.95,
					"close": 57.45,
					"volume": 2613142
				}, ...
			}
		}, ...
	]
}
```


# Create and update quote to mysql database
```
./data-retriever-cli update <mysql_ip> <db_name> <mysql_username> <mysql_password> -i <update_interval_in_seconds>
```
(p.s. If the table 'stocks_quotes' does not exist, it will create one)
![Screenshot](https://raw.githubusercontent.com/chanyk-joseph/dissertation/master/stock/data-retriever-cli/mysql_sample_content.PNG)
