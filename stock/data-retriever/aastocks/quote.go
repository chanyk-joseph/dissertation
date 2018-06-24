package aastocks

import (
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/pkg/errors"
)

type EquityQuote struct {
	Symbol string `json:"symbol"`

	LastTradedPrice float64 `json:"last_traded_price"`
	Open            float64 `json:"open"`
	Low             float64 `json:"low"`
	High            float64 `json:"high"`
	Bid             float64 `json:"bid"`
	Ask             float64 `json:"ask"`
	PreviousClose   float64 `json:"privious_close"`
	Volume          string  `json:"volume"`

	LotSize        int     `json:"lot_size"`
	Turnover       string  `json:"turnover"`
	PE             float64 `json:"pe"`
	Yield          string  `json:"yield"`
	DividendPayout string  `json:"dividend_payout"`
	EPS            float64 `json:"eps"`
	MarketCapital  string  `json:"market_capital"`
	NetAssetValue  float64 `json:"net_asset_value"`
	Low52Weeks     float64 `json:"low_52_weeks"`
	High52Weeks    float64 `json:"high_52_weeks"`
}

func (quote EquityQuote) ToJSONString() string {
	return utils.ObjectToJSONString(quote)
}

// Quote From aastocks (not working at this moment)
// Example symbol: 00700
func Quote(standardSymbol models.StandardSymbol) (EquityQuote, error) {
	result := EquityQuote{}

	symbol := standardSymbol.Symbol[:len(standardSymbol.Symbol)-3]
	urlStr := "http://www.aastocks.com/en/mobile/Quote.aspx?symbol=" + symbol

	_, bodyString, err := utils.HttpGetResponseContent(urlStr)
	if err != nil {
		return result, err
	}

	var re = regexp.MustCompile(`(?m)<div class="text_last"[\s\S]*?<span.*?>([0-9.]+?)<\/span>[\s\S]*?L\/H ([0-9.]+?)-([0-9.]+)[\s\S]*?Bid[\s\S]*?>([0-9.]+?)<[\s\S]*?Ask[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.KMB]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.KMB]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.%]+?)<[\s\S]*?>([0-9.%]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?>([0-9.,KMB]+?)<[\s\S]*?>([0-9.]+?)<[\s\S]*?([0-9.]+?) - ([0-9.]+)[\s\S]*?stockid=([0-9A-Z.]+)`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 20 {
		return result, errors.Errorf("Unable To Extract aastocks Quote: \n%s", bodyString)
	}
	result.LastTradedPrice = utils.StringToFloat64(match[1])
	result.Low = utils.StringToFloat64(match[2])
	result.High = utils.StringToFloat64(match[3])
	result.Bid = utils.StringToFloat64(match[4])
	result.Ask = utils.StringToFloat64(match[5])
	result.Open = utils.StringToFloat64(match[6])
	result.PreviousClose = utils.StringToFloat64(match[7])
	result.Volume = match[8]
	result.LotSize = utils.StringToInt(match[9])
	result.Turnover = match[10]
	result.PE = utils.StringToFloat64(match[11])
	result.Yield = match[12]
	result.DividendPayout = match[13]
	result.EPS = utils.StringToFloat64(match[14])
	result.MarketCapital = match[15]
	result.NetAssetValue = utils.StringToFloat64(match[16])
	result.Low52Weeks = utils.StringToFloat64(match[17])
	result.High52Weeks = utils.StringToFloat64(match[18])
	result.Symbol = match[19]

	return result, nil
}

func ToStandardQuote(q EquityQuote) models.StandardQuote {
	tmp := models.StandardQuote{}
	tmp.Open = q.Open
	tmp.Low = q.Low
	tmp.High = q.High
	tmp.Close = q.LastTradedPrice
	tmp.Volume = int64(utils.ConvertNumberWithUnitToActualNumber(q.Volume))

	return tmp
}

/*
Example JSON:
http://www.aastocks.com/en/mobile/Quote.aspx?symbol=00700
{
	"symbol": "00700.HK",
	"last_traded_price": 419,
	"open": 420,
	"low": 415.6,
	"high": 421,
	"bid": 418.8,
	"ask": 419,
	"privious_close": 415,
	"volume": "15.48M",
	"lot_size": 100,
	"turnover": "6.49B",
	"pe": 46,
	"yield": "0.21%",
	"dividend_payout": "9.661%",
	"eps": 9.109,
	"market_capital": "3,982.05B",
	"net_asset_value": 32.32,
	"low_52_weeks": 260.4,
	"high_52_weeks": 476.6
}
*/
