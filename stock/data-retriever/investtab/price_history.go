package investtab

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
)

type PriceRecord struct {
	MDEntryTimeInMS int64     `json:"mdEntryTime"`
	Date            time.Time `json:"date"`

	FirstPrice        string `json:"firstPx"`
	LowPrice          string `json:"lowPx"`
	HighPrice         string `json:"highPx"`
	LastPrice         string `json:"price"`
	TotalVolumeTraded string `json:"totalVolumeTraded"`
}

func (priceRecord PriceRecord) ToJSONString() string {
	return utils.ObjectToJSONString(priceRecord)
}

func GetPriceRecords(standardSymbol models.StandardSymbol, startTime time.Time, endTime time.Time) ([]PriceRecord, error) {
	result := []PriceRecord{}
	symbol := standardSymbol.Symbol[:len(standardSymbol.Symbol)-3]

	startTimeStr := fmt.Sprintf("%d%02d%02d", startTime.Year(), startTime.Month(), startTime.Day())
	endTimeStr := fmt.Sprintf("%d%02d%02d", endTime.Year(), endTime.Month(), endTime.Day())
	urlStr := fmt.Sprintf("https://api.investtab.com/api/quote/trubuzz/histories/date-range?country=HK&exchange=HKSE&symbol=%s&period=d1&begin=%s&end=%s", symbol, startTimeStr, endTimeStr)

	headers := map[string]string{
		"Authorization": "Basic ZmFrZUBnbWFpbC5jb206dGVzdGFwaQ==",
	}
	_, bodyStr, err := utils.HttpGetResponseContentWithHeaders(urlStr, headers)
	if err != nil {
		return result, err
	}
	if err = json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return result, err
	}

	for i := range result {
		record := &result[i]
		record.Date = time.Unix(record.MDEntryTimeInMS/1000, 0)
	}

	return result, nil
}

/*
Example JSON:
https://api.investtab.com/api/quote/trubuzz/histories/date-range?country=HK&exchange=HKSE&symbol=00001&period=d1&end=20180617&begin=20100622
[
	{
        "firstPx": "117.70",
        "mdEntryTime": 1298505600000,
        "price": "116.60",
        "lowPx": "116.40",
        "highPx": "119.20",
        "totalVolumeTraded": "4388181"
    },
    {
        "firstPx": "116.90",
        "mdEntryTime": 1298592000000,
        "price": "119.90",
        "lowPx": "116.90",
        "highPx": "120.40",
        "totalVolumeTraded": "4977125"
    },
    {
        "firstPx": "118.30",
        "mdEntryTime": 1298851200000,
        "price": "121.20",
        "lowPx": "118.00",
        "highPx": "122.40",
        "totalVolumeTraded": "5249555"
    }
]
*/
