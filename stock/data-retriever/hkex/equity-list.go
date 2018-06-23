package hkex

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/pkg/errors"
)

type Stock struct {
	ShortCompanyName      string `json:"nm"`
	Symbol                string `json:"sym"`
	ReutersInstrumentCode string `json:"ric"`
}

func (stock Stock) ToJSONString() string {
	return utils.ObjectToJSONString(stock)
}

func GetEquityList() ([]Stock, error) {
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
		return nil, errors.Errorf("Failed To Get HKEX Equity List | %s | Response Status Code: %v", urlStr, resp.StatusCode)
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

/*
Sample Response:
{
	"data": {
		"responsecode": "000",
		"responsemsg": "",
		"lastupd": "08 Jun 2018 16:30 HKT",
		"lastupd_l": 1528445703000,
		"stocklist": [{
				"asset_subtype": "ODSH",
				"sym": "700",
				"mktcap": "3,944.19",
				"ls": "415.000",
				"am": "12.08",
				"ric": "0700.HK",
				"product_subtype": null,
				"am_u": "B",
				"yld": "0.21",
				"pc": "-3.31",
				"pe": "44.10",
				"nc": "-14.200",
				"ccy": "HKD",
				"nm": "騰訊控股",
				"mktcap_u": "B"
			}, {
				"asset_subtype": "ODSH",
				"sym": "2318",
				"mktcap": "584.63",
				"ls": "78.500",
				"am": "3.15",
				"ric": "2318.HK",
				"product_subtype": null,
				"am_u": "B",
				"yld": "2.63",
				"pc": "-1.32",
				"pe": "12.69",
				"nc": "-1.050",
				"ccy": "HKD",
				"nm": "中國平安",
				"mktcap_u": "B"
			}
		]
	},
	"qid": "1528571904740"
}
*/
