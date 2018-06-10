package aastocks

import (
	"io/ioutil"
	"net/http"
	"regexp"

	"../common/util"
	"github.com/pkg/errors"
)

type EquityQuote struct {
	LastTradedPrice float32 `json:"last_traded_price"`
	Low             float32 `json:"low"`
	High            float32 `json:"high"`
	Volume          string  `json:"volume"`
	Turnover        string  `json:"turnover"`
	PE              float32 `json:"pe"`
	LotSize         int     `json:"lot_size"`
	MarketCapital   string  `json:"market_capital"`
	EPS             float32 `json:"eps"`
	Yield           string  `json:"yield"`
	Low52Weeks      float32 `json:"low_52_weeks"`
	High52Weeks     float32 `json:"high_52_weeks"`
}

func (quote EquityQuote) ToJSONString() string {
	return util.ObjectToJsonString(quote)
}

// Quote From aastocks (not working at this moment)
// Example symbol: 00700
func Quote(symbol string) (EquityQuote, error) {
	result := EquityQuote{}

	urlStr := "http://www.aastocks.com/en/ltp/rtquote.aspx?symbol=" + symbol

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

	var re = regexp.MustCompile(`(?m)<table class="tb-c" cellspacing="1" cellpadding="1" width="357">[\s\S]*?<span class=.*?>([0-9.]+?)<[\s\S]*?<strong>([0-9.]+?) - ([0-9.]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?([0-9.,%KMB]+?)</strong>[\s\S]*?([0-9.,KMB]+?)</strong>[\s\S]*?<strong>[\s\S]*?([0-9.]+?) - ([0-9.]+?)</strong>`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 13 {
		return result, errors.Errorf("Unable To Extract aastocks Quote: \n%s", bodyString)
	}
	result.LastTradedPrice = util.StringToFloat32(match[1])
	result.Low = util.StringToFloat32(match[2])
	result.High = util.StringToFloat32(match[3])
	result.Volume = match[4]
	result.MarketCapital = match[5]
	result.Turnover = match[6]
	result.EPS = util.StringToFloat32(match[7])
	result.PE = util.StringToFloat32(match[8])
	result.Yield = match[9]
	result.LotSize = util.StringToInt(match[10])
	result.Low52Weeks = util.StringToFloat32(match[11])
	result.High52Weeks = util.StringToFloat32(match[12])

	return result, nil
}
