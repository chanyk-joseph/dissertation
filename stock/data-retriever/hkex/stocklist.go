package hkex

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"regexp"

	"github.com/pkg/errors"
)

type Stock struct {
	ShortCompanyName      string `json:"nm"`
	Symbol                string `json:"sym"`
	ReutersInstrumentCode string `json:"ric"`
}

type EquityQuote struct {
	UpdateTime string `json:"db_updatetime"`

	ShortCompanyName      string `json:"nm_s"`
	CompanyName           string `json:"nm"`
	Symbol                string `json:"sym"`
	ReutersInstrumentCode string `json:"ric"`

	EPS               float32 `json:"eps"`
	EPS_Currency      string  `json:"eps_ccy"`
	PE                string  `json:"pe"`
	DividendYield     string  `json:"div_yield"`
	MarketCapital     string  `json:"mkt_cap"`
	MarketCapitalUnit string  `json:"mkt_cap_u"`

	LastPrice     string `json:"ls"`
	PreviousClose string `json:"hc"`
	Open          string `json:"op"`
	High          string `json:"hi"`
	Low           string `json:"lo"`
	High52Week    string `json:"hi52"`
	Low52Week     string `json:"lo52"`

	TurnOver     string `json:"am"`
	TurnOverUnit string `json:"am_u"`
	Volumn       string `json:"vo"`
	VolumnUnit   string `json:"vo_u"`
	Bid          string `json:"bd"`
	Ask          string `json:"as"`
}

func (stock Stock) ToString() string {
	buf, err := json.MarshalIndent(stock, "", "	")
	if err != nil {
		panic(err)
	}

	return string(buf)
}

func (quote EquityQuote) ToString() string {
	buf, err := json.MarshalIndent(quote, "", "	")
	if err != nil {
		panic(err)
	}

	return string(buf)
}

func getAccessToken() (string, error) {
	urlStr := "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=zh-hk"

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil)
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")

	resp, err := client.Do(r)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", errors.Errorf("Failed To Get HKEX Access Token | %s | Response Status Code: %v", urlStr, resp.StatusCode)
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	re := regexp.MustCompile(`(?m)Base64-AES-Encrypted-Token"[\s\S]*?"(.*?)";`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return "", errors.New("Unable To Locate Access Key From Response: \n" + bodyString)
	}

	return match[1], nil
}

func GetStockList() ([]Stock, error) {
	accessToken, err := getAccessToken()
	if err != nil {
		return nil, err
	}
	urlStr := "https://www1.hkex.com.hk/hkexwidget/data/getequityfilter?lang=chi&token=" + accessToken + "&sort=5&order=0&all=1&qid=1528566852598&callback=jQuery3110021454077880299405_1528566851996&_=1528566851998"

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil) // URL-encoded payload
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")
	r.Header.Add("Referer", "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=zh-hk")

	resp, err := client.Do(r)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, errors.Errorf("Failed To Get HKEX Access Token | %s | Response Status Code: %v", urlStr, resp.StatusCode)
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	re := regexp.MustCompile(`(?m)\(([\s\S]*?)\)$`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return nil, errors.New("Unable To Locate Stock List Array From Response: \n" + bodyString)
	}
	jsonStr := match[1]

	stockListResp := &struct {
		Data struct {
			StockList []Stock
		}
	}{}
	err = json.Unmarshal([]byte(jsonStr), &stockListResp)
	if err != nil {
		return nil, err
	}

	var stocks []Stock
	stocks = append(stocks, stockListResp.Data.StockList...)

	return stocks, nil
}

func Quote(sym string) (EquityQuote, error) {
	result := EquityQuote{}

	accessToken, err := getAccessToken()
	if err != nil {
		return result, err
	}
	urlStr := "https://www1.hkex.com.hk/hkexwidget/data/getequityquote?sym=" + sym + "&token=" + accessToken + "&lang=chi&qid=1528572605481&callback=jQuery311037427382333777826_1528572604782&_=1528572604783"

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil) // URL-encoded payload
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")
	r.Header.Add("Referer", "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=zh-hk")

	resp, err := client.Do(r)
	if err != nil {
		return result, err
	}
	if resp.StatusCode != http.StatusOK {
		return result, errors.Errorf("Failed To Get HKEX Access Token | %s | Response Status Code: %v", urlStr, resp.StatusCode)
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return result, errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	re := regexp.MustCompile(`(?m)\(([\s\S]*?)\)$`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return result, errors.New("Unable To Locate Stock List Array From Response: \n" + bodyString)
	}
	jsonStr := match[1]

	quoteResp := &struct {
		Data struct {
			Quote EquityQuote
		}
	}{}

	err = json.Unmarshal([]byte(jsonStr), &quoteResp)
	if err != nil {
		return result, err
	}

	result = quoteResp.Data.Quote
	return result, nil
}
