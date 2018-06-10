package bloomberg

import (
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"

	"../common/util"
	"github.com/pkg/errors"
)

type EquityQuote struct {
	Code        string `json:"code"`
	CompanyName string `json:"company_name"`
	Exchange    string `json:"exchange"`
	MarketCap   string `json:"market_cap"`
	Currency    string `json:"currency"`

	PreviousClose   float32 `json:"previous_close"`
	Open            float32 `json:"open"`
	Low             float32 `json:"low"`
	High            float32 `json:"high"`
	LastTradedPrice float32 `json:"last_traded_price"`
	Volumn          int     `json:"volumn"`

	Low52Weeks  float32 `json:"low_52_weeks"`
	High52Weeks float32 `json:"high_52_weeks"`

	PE                float32 `json:"PE"`
	BestPE            float32 `json:"best_PE"`
	BestPEG           float32 `json:"best_PEG"`
	SharesOutstanding string  `json:"shares_outstanding"`
	PriceToBookRatio  float32 `json:"price_to_book_ratio"`
	PriceToSalesRatio float32 `json:"price_to_sales_ratio"`

	OneYearReturn        string  `json:"one_year_return"`
	AverageVolumn30Days  int     `json:"average_volumn_30_days"`
	EPS                  float32 `json:"EPS"`
	BestEPSInCurrentYear float32 `json:"best_EPS_in_current_year"`
	Dividend             string  `json:"dividend"`
	LastDividendReported float32 `json:"last_dividend_reported"`
}

func (quote EquityQuote) ToJSONString() string {
	return util.ObjectToJsonString(quote)
}

// Quote return result from bloomberg
// Example Code: 700:HK
func Quote(code string) (EquityQuote, error) {
	result := EquityQuote{}
	urlStr := "https://www.bloomberg.com/quote/" + code

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil) // URL-encoded payload
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")

	resp, err := client.Do(r)
	if err != nil {
		return result, err
	}
	if resp.StatusCode != http.StatusOK {
		return result, errors.Errorf("Failed To Get Quote | Response Status Code: %v | Request: \n%s", resp.StatusCode, util.FormatRequest(r))
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return result, errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	var re = regexp.MustCompile(`(?m)companyId.*?>(.*?)<.*?exchange.*?>(.*?)<.*?companyName.*?>(.*?)<.*?priceText.*?>(.*?)<.*?currency.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?value.*?>(.*?)<.*?textLeft.*?>(.*?)<.*?-.*?textRight.*?>(.*?)<.*?textLeft.*?>(.*?)<.*?-.*?textRight.*?>(.*?)<`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 14 {
		return result, errors.Errorf("Unable To Extract Bloomberg Quote: \n%s", bodyString)
	}
	result.Code = match[1]
	result.Exchange = match[2]
	result.CompanyName = match[3]
	result.LastTradedPrice = util.StringToFloat32(match[4])
	result.Currency = match[5]
	result.Open = util.StringToFloat32(match[6])
	result.PreviousClose = util.StringToFloat32(match[7])
	result.Volumn = util.StringToInt(strings.Replace(match[8], ",", "", -1))
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
	result.AverageVolumn30Days = util.StringToInt(strings.Replace(match[8], ",", "", -1))
	result.EPS = util.StringToFloat32(match[9])
	result.BestEPSInCurrentYear = util.StringToFloat32(match[10])
	result.Dividend = match[11]
	result.LastDividendReported = util.StringToFloat32(match[12])

	return result, nil
}
