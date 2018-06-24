Serve stock quote via RestAPI / save to mysql database

## Stock Quote Rest API Usage
```
./data-retriever-cli web -p 8888
```
http://127.0.0.1:8888/quote/700 (p.s. the stock symbol could be in many formats: 700, 0700, 00700, 700.HK, 700:HK etc.)
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
http://127.0.0.1:8888/hsicomponents
```json
["00941.HK","01038.HK","01044.HK","01088.HK","01093.HK","01109.HK","01113.HK","01299.HK","01398.HK","01928.HK","01997.HK","02007.HK","02018.HK","02318.HK","02319.HK","02382.HK","02388.HK","02628.HK","03328.HK","03988.HK","00001.HK","00002.HK","00003.HK","00005.HK","00006.HK","00011.HK","00012.HK","00016.HK","00017.HK","00019.HK","00023.HK","00027.HK","00066.HK","00083.HK","00101.HK","00144.HK","00151.HK","00175.HK","00267.HK","00288.HK","00386.HK","00388.HK","00688.HK","00700.HK","00762.HK","00823.HK","00836.HK","00857.HK","00883.HK","00939.HK"]
```
http://127.0.0.1:8888/hsicomponents/quote
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
...
				"hkex": {
					"open": 70,
					"low": 68.3,
					"high": 70,
					"close": 68.7,
					"volume": 15410000
				}
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
...
				"investtab": {
					"open": 57.9,
					"low": 57,
					"high": 57.95,
					"close": 57.45,
					"volume": 2613142
				}
			}
		}, 
...
		{
			"symbol": "00939.HK",
			"quotes": {
				"aastocks": {
					"open": 7.39,
					"low": 7.33,
					"high": 7.42,
					"close": 7.38,
					"volume": 408770000
				},
...
				"investtab": {
					"open": 7.39,
					"low": 7.33,
					"high": 7.42,
					"close": 7.38,
					"volume": 408772500
				}
			}
		}
	]
}
```

------------------

## Create and update quote to mysql database
![Screenshot](https://raw.githubusercontent.com/chanyk-joseph/dissertation/stock/data-retriever-cli/mysql_sample_content.PNG)