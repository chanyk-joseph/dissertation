package investtab

import (
	"encoding/json"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
)

type EquityQuote struct {
	Symbol     string `json:"symbol"`
	UpdateTime string `json:"as_of_date"`

	Open  float64 `json:"open"`
	Low   float64 `json:"low"`
	High  float64 `json:"high"`
	Close float64 `json:"close"`

	Volume          float64 `json:"volume"`
	VolumeAvg20Days float64 `json:"volume_avg_20d"`

	Low10Days   float64 `json:"low_10d"`
	Low250Days  float64 `json:"low_250d"`
	High10Days  float64 `json:"high_10d"`
	High250Days float64 `json:"high_250d"`

	SMA10  float64 `json:"sma10"`
	SMA20  float64 `json:"sma20"`
	SMA50  float64 `json:"sma50"`
	SMA100 float64 `json:"sma100"`
	SMA250 float64 `json:"sma250"`

	BBandsLower float64 `json:"bbands_lower"`
	BBandsUpper float64 `json:"bbands_upper"`

	DIMinus float64 `json:"di_minus"`
	DIPlus  float64 `json:"di_plus"`

	MACD       float64 `json:"macd"`
	MACDSignal float64 `json:"macd_signal"`
	MACDHist   float64 `json:"macd_hist"`

	StcK float64 `json:"stc_k"`
	StcD float64 `json:"stc_d"`

	ADX float64 `json:"adx"`
	RSI float64 `json:"rsi"`

	PricesGap float64 `json:"prices_gap"`

	ChangeFromOpen             float64 `json:"change_from_open"`
	ChangeFromOpenPCT          float64 `json:"change_from_open_pct"`
	ChangeFromPreviousClose    float64 `json:"change_from_prev_close"`
	ChangeFromPreviousClosePCT float64 `json:"change_from_prev_close_pct"`
}

func (quote EquityQuote) ToJSONString() string {
	return utils.ObjectToJSONString(quote)
}

func Quote(standardSymbol models.StandardSymbol) (EquityQuote, error) {
	var result EquityQuote
	symbol := strings.Replace(standardSymbol.Symbol, ".", ":", -1)

	urlStr := "https://api.investtab.com/api/quote/" + symbol + "/technical"
	headers := map[string]string{
		"Accept":  "application/json, text/plain, */*",
		"Referer": "https://www.investtab.com/en/filter",
	}
	_, bodyStr, err := utils.HttpGetResponseContentWithHeaders(urlStr, headers)
	if err != nil {
		return result, err
	}

	err = json.Unmarshal([]byte(bodyStr), &result)
	if err != nil {
		return result, err
	}

	return result, nil
}

func ToStandardQuote(q EquityQuote) models.StandardQuote {
	tmp := models.StandardQuote{}
	tmp.Open = q.Open
	tmp.Low = q.Low
	tmp.High = q.High
	tmp.Close = q.Close
	tmp.Volume = int64(q.Volume)

	return tmp
}

/*
Example JSON:
https://api.investtab.com/api/quote/00001:HK/technical
{
	"low_10d": 87.65,
	"sma100": 95.3395,
	"as_of_date": "2018-06-13T00:00:00",
	"sma250": 97.9654,
	"close": 89.5,
	"open": 91.5,
	"bbands_upper": 91.3828,
	"di_plus": 25.7489,
	"adx": 14.5943,
	"prices_gap": null,
	"low": 89.35,
	"bbands_lower": 87.5072,
	"sma20": 89.445,
	"change_from_open_pct": -2.1858,
	"volume_avg_20d": 5774078.05,
	"high_10d": 92.0,
	"rsi": 45.7161,
	"symbol": "00001:HK",
	"volume": 5289102.0,
	"macd": -0.2653,
	"high_250d": 108.9,
	"macd_signal": -0.5684,
	"macd_hist": 0.3031,
	"di_minus": 29.3288,
	"high": 91.5,
	"sma10": 90.0,
	"change_from_open": -2.0,
	"sma50": 91.139,
	"low_250d": 87.1,
	"stc_k": 74.1848,
	"stc_d": 78.3694,
	"change_from_prev_close_pct": -2.2392,
	"change_from_prev_close": -2.05
}

{
	"low_10d": 115.4,
	"sma100": 135.247,
	"as_of_date": "2018-06-13T00:00:00",
	"sma250": 133.141,
	"close": 121.6,
	"open": 125.0,
	"bbands_upper": 131.6314,
	"di_plus": 28.6915,
	"adx": 17.0233,
	"prices_gap": -0.6334,
	"low": 120.4,
	"bbands_lower": 108.1486,
	"sma20": 119.89,
	"change_from_open_pct": -2.72,
	"volume_avg_20d": 7534124.0,
	"high_10d": 134.6,
	"rsi": 49.1626,
	"symbol": "02018:HK",
	"volume": 8945350.0,
	"macd": 1.2892,
	"high_250d": 185.0,
	"macd_signal": 0.0695,
	"macd_hist": 1.2197,
	"di_minus": 25.8411,
	"high": 125.5,
	"sma10": 124.09,
	"change_from_open": -3.4,
	"sma50": 123.604,
	"low_250d": 94.05,
	"stc_k": 59.4395,
	"stc_d": 67.1019,
	"change_from_prev_close_pct": -4.1009,
	"change_from_prev_close": -5.2
}
*/
