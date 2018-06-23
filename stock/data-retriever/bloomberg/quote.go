package bloomberg

import (
	"regexp"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/pkg/errors"
)

type EquityQuote struct {
	Symbol      string `json:"symbol"`
	CompanyName string `json:"company_name"`
	Exchange    string `json:"exchange"`
	MarketCap   string `json:"market_cap"`
	Currency    string `json:"currency"`

	PreviousClose   float64 `json:"previous_close"`
	Open            float64 `json:"open"`
	Low             float64 `json:"low"`
	High            float64 `json:"high"`
	LastTradedPrice float64 `json:"last_traded_price"`
	Volume          int     `json:"volume"`

	Low52Weeks  float64 `json:"low_52_weeks"`
	High52Weeks float64 `json:"high_52_weeks"`

	PE                float64 `json:"PE"`
	BestPE            float64 `json:"best_PE"`
	BestPEG           float64 `json:"best_PEG"`
	SharesOutstanding string  `json:"shares_outstanding"`
	PriceToBookRatio  float64 `json:"price_to_book_ratio"`
	PriceToSalesRatio float64 `json:"price_to_sales_ratio"`

	OneYearReturn        string  `json:"one_year_return"`
	AverageVolume30Days  int     `json:"average_volume_30_days"`
	EPS                  float64 `json:"EPS"`
	BestEPSInCurrentYear float64 `json:"best_EPS_in_current_year"`
	Dividend             string  `json:"dividend"`
	LastDividendReported float64 `json:"last_dividend_reported"`
}

func (quote EquityQuote) ToJSONString() string {
	return util.ObjectToJSONString(quote)
}

// Quote return result from bloomberg
// Example symbol: 700:HK
func Quote(standardSymbol models.StandardSymbol) (EquityQuote, error) {
	result := EquityQuote{}

	code, err := util.ExtractStockCode(standardSymbol.Symbol)
	if err != nil {
		return result, err
	}
	symbol := code + ":HK"
	urlStr := "https://www.bloomberg.com/quote/" + symbol

	_, bodyString, err := util.HttpGetResponseContent(urlStr)
	if err != nil {
		return result, err
	}

	var re = regexp.MustCompile(`(?m)companyId.*?>(.*?)<.*?exchange.*?>(.*?)<.*?companyName.*?>(.*?)<.*?priceText.*?>(.*?)<.*?currency.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?textLeft.*?>(.*?)<.*?-.*?textRight.*?>(.*?)<.*?textLeft.*?>(.*?)<.*?-.*?textRight.*?>(.*?)<`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 14 {
		return result, errors.Errorf("Unable To Extract Bloomberg Quote: \n%s", bodyString)
	}
	result.Symbol = match[1]
	result.Exchange = match[2]
	result.CompanyName = match[3]
	result.LastTradedPrice = util.StringToFloat64(match[4])
	result.Currency = match[5]
	result.Open = util.StringToFloat64(match[6])
	result.PreviousClose = util.StringToFloat64(match[7])
	result.Volume = util.StringToInt(strings.Replace(match[8], ",", "", -1))
	result.MarketCap = match[9]
	result.Low = util.StringToFloat64(match[10])
	result.High = util.StringToFloat64(match[11])
	result.Low52Weeks = util.StringToFloat64(match[12])
	result.High52Weeks = util.StringToFloat64(match[13])

	re = regexp.MustCompile(`(?m)<span>P/E Ratio<\/span>.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<`)
	match = re.FindStringSubmatch(bodyString)
	if len(match) != 13 {
		return result, errors.Errorf("Unable To Extract Bloomberg Quote: \n%s", bodyString)
	}
	result.PE = util.StringToFloat64(match[1])
	result.BestPE = util.StringToFloat64(match[2])
	result.BestPEG = util.StringToFloat64(match[3])
	result.SharesOutstanding = match[4]
	result.PriceToBookRatio = util.StringToFloat64(match[5])
	result.PriceToSalesRatio = util.StringToFloat64(match[6])
	result.OneYearReturn = match[7]
	result.AverageVolume30Days = util.StringToInt(strings.Replace(match[8], ",", "", -1))
	result.EPS = util.StringToFloat64(match[9])
	result.BestEPSInCurrentYear = util.StringToFloat64(match[10])
	result.Dividend = match[11]
	result.LastDividendReported = util.StringToFloat64(match[12])

	return result, nil
}

/*
Example JSON:
https://www.bloomberg.com/quote/700:HK
{
	"symbol": "700:HK",
	"company_name": "Tencent Holdings Ltd",
	"exchange": "Hong Kong",
	"market_cap": "3.982T",
	"currency": "HKD",
	"previous_close": 415,
	"open": 420,
	"low": 415.6,
	"high": 421,
	"last_traded_price": 419,
	"volume": 15480976,
	"low_52_weeks": 260.4,
	"high_52_weeks": 476.6,
	"PE": 40.09,
	"best_PE": 36.1544,
	"best_PEG": 1.5491,
	"shares_outstanding": "9.5B",
	"price_to_book_ratio": 11.7053,
	"price_to_sales_ratio": 12.3025,
	"one_year_return": "51.37%",
	"average_volume_30_days": 25278900,
	"EPS": 8.53,
	"best_EPS_in_current_year": 8.873,
	"dividend": "0.21%",
	"last_dividend_reported": 0.88
}
*/
