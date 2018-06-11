package bloomberg

import (
	"regexp"
	"strings"

	"../common/util"
	"github.com/pkg/errors"
)

/*
Example JSON:
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
type EquityQuote struct {
	Symbol      string `json:"symbol"`
	CompanyName string `json:"company_name"`
	Exchange    string `json:"exchange"`
	MarketCap   string `json:"market_cap"`
	Currency    string `json:"currency"`

	PreviousClose   float32 `json:"previous_close"`
	Open            float32 `json:"open"`
	Low             float32 `json:"low"`
	High            float32 `json:"high"`
	LastTradedPrice float32 `json:"last_traded_price"`
	Volume          int     `json:"volume"`

	Low52Weeks  float32 `json:"low_52_weeks"`
	High52Weeks float32 `json:"high_52_weeks"`

	PE                float32 `json:"PE"`
	BestPE            float32 `json:"best_PE"`
	BestPEG           float32 `json:"best_PEG"`
	SharesOutstanding string  `json:"shares_outstanding"`
	PriceToBookRatio  float32 `json:"price_to_book_ratio"`
	PriceToSalesRatio float32 `json:"price_to_sales_ratio"`

	OneYearReturn        string  `json:"one_year_return"`
	AverageVolume30Days  int     `json:"average_volume_30_days"`
	EPS                  float32 `json:"EPS"`
	BestEPSInCurrentYear float32 `json:"best_EPS_in_current_year"`
	Dividend             string  `json:"dividend"`
	LastDividendReported float32 `json:"last_dividend_reported"`
}

func (quote EquityQuote) ToJSONString() string {
	return util.ObjectToJsonString(quote)
}

// Quote return result from bloomberg
// Example symbol: 700:HK
func Quote(symbol string) (EquityQuote, error) {
	result := EquityQuote{}
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
	result.LastTradedPrice = util.StringToFloat32(match[4])
	result.Currency = match[5]
	result.Open = util.StringToFloat32(match[6])
	result.PreviousClose = util.StringToFloat32(match[7])
	result.Volume = util.StringToInt(strings.Replace(match[8], ",", "", -1))
	result.MarketCap = match[9]
	result.Low = util.StringToFloat32(match[10])
	result.High = util.StringToFloat32(match[11])
	result.Low52Weeks = util.StringToFloat32(match[12])
	result.High52Weeks = util.StringToFloat32(match[13])

	re = regexp.MustCompile(`(?m)<span>P/E Ratio<\/span>.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<.*?fieldValue.*?>(.*?)<`)
	match = re.FindStringSubmatch(bodyString)
	if len(match) != 13 {
		return result, errors.Errorf("Unable To Extract Bloomberg Quote: \n%s", bodyString)
	}
	result.PE = util.StringToFloat32(match[1])
	result.BestPE = util.StringToFloat32(match[2])
	result.BestPEG = util.StringToFloat32(match[3])
	result.SharesOutstanding = match[4]
	result.PriceToBookRatio = util.StringToFloat32(match[5])
	result.PriceToSalesRatio = util.StringToFloat32(match[6])
	result.OneYearReturn = match[7]
	result.AverageVolume30Days = util.StringToInt(strings.Replace(match[8], ",", "", -1))
	result.EPS = util.StringToFloat32(match[9])
	result.BestEPSInCurrentYear = util.StringToFloat32(match[10])
	result.Dividend = match[11]
	result.LastDividendReported = util.StringToFloat32(match[12])

	return result, nil
}
