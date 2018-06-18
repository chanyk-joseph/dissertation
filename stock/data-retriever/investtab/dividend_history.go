package investtab

import (
	"encoding/json"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/converter"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
)

type DividendRecord struct {
	ID     int    `json:"id"`
	Symbol string `json:"symbol"`

	Period        string    `json:"period"`
	PeriodLoc     PeriodLoc `json:"period_loc"`
	Particular    string    `json:"particular"`
	ParticularLoc string    `json:"particular_loc"`
	CAType        string    `json:"ca_type"`

	AsOfDate       string `json:"as_of_date"`
	RecordDate     string `json:"record_date"`
	AnnounceDate   string `json:"announce_date"`
	ExDividendDate string `json:"ex_date"` //除淨日
	PayDate        string `json:"pay_date"`
}

type PeriodLoc struct {
	SimplifiedChinese  string `json:"sc"`
	TraditionalChinese string `json:"tc"`
	English            string `json:"en"`
}

func (dividendRecord DividendRecord) ToJSONString() string {
	return util.ObjectToJSONString(dividendRecord)
}

func GetDividendRecords(standardSymbol converter.StandardSymbol) ([]DividendRecord, error) {
	result := []DividendRecord{}
	symbol := strings.Replace(standardSymbol.Symbol, ".", ":", -1)

	urlStr := "https://api.investtab.com/api/quote/" + symbol + "/dividend-history"
	_, bodyStr, err := util.HttpGetResponseContent(urlStr)
	if err != nil {
		return result, err
	}
	if err = json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return result, err
	}

	return result, nil
}

/*
Example JSON: https://api.investtab.com/api/quote/00001:HK/dividend-history
[
    {
        "symbol": "00001:HK",
        "book_close_period": "",
        "period": "Final",
        "period_loc": {
            "sc": "末期业绩",
            "en": "Final",
            "tc": "末期業績"
        },
        "announce_date": "2018-03-16T00:00:00",
        "as_of_date": "2017-12-31T00:00:00",
        "particular_loc": "股息: 港元 2.07",
        "pay_date": "2018-05-31T00:00:00",
        "particular": "Cash Dividend: HKD 2.07",
        "ca_type": "CD",
        "ex_date": "2018-05-15T00:00:00",
        "id": 87387,
        "record_date": "2018-05-16T00:00:00"
    },
    {
        "symbol": "00001:HK",
        "book_close_period": "",
        "period": "Interim",
        "period_loc": {
            "sc": "中期业绩",
            "en": "Interim",
            "tc": "中期業績"
        },
        "announce_date": "2017-08-03T00:00:00",
        "as_of_date": "2017-12-31T00:00:00",
        "particular_loc": "股息: 港元 0.78",
        "pay_date": "2017-09-14T00:00:00",
        "particular": "Cash Dividend: HKD 0.78",
        "ca_type": "CD",
        "ex_date": "2017-09-04T00:00:00",
        "id": 84094,
        "record_date": "2017-09-05T00:00:00"
    },
    {
        "symbol": "00001:HK",
        "book_close_period": "2004/05/13-2004/05/20",
        "period": "Final",
        "period_loc": {
            "sc": "末期业绩",
            "en": "Final",
            "tc": "末期業績"
        },
        "announce_date": "2004-03-18T00:00:00",
        "as_of_date": "2003-12-31T00:00:00",
        "particular_loc": "股息: 港元 1.3",
        "pay_date": "2004-05-25T00:00:00",
        "particular": "Cash Dividend: HKD 1.3",
        "ca_type": "CD",
        "ex_date": "2004-05-11T00:00:00",
        "id": 37201,
        "record_date": null
    }
]
*/
