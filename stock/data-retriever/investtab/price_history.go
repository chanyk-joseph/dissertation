package investtab

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
)

type PriceRecord struct {
	MDEntryTimeInMS int64 `json:"mdEntryTime"`
	MDEntryTime     time.Time

	FirstPrice        string `json:"firstPx"`
	LowPrice          string `json:"lowPx"`
	HighPrice         string `json:"highPx"`
	LastPrice         string `json:"price"`
	TotalVolumeTraded string `json:"totalVolumeTraded"`
}

func (priceRecord PriceRecord) ToJSONString() string {
	return util.ObjectToJSONString(priceRecord)
}

func GetPriceRecords(symbol string, startTime time.Time, endTime time.Time) ([]PriceRecord, error) {
	result := []PriceRecord{}

	startTimeStr := fmt.Sprintf("%d%02d%02d", startTime.Year(), startTime.Month(), startTime.Day())
	endTimeStr := fmt.Sprintf("%d%02d%02d", endTime.Year(), endTime.Month(), endTime.Day())
	urlStr := fmt.Sprintf("https://api.investtab.com/api/quote/trubuzz/histories/date-range?country=HK&exchange=HKSE&symbol=%s&period=d1&begin=%s&end=%s", symbol, startTimeStr, endTimeStr)

	fmt.Println(urlStr)
	headers := map[string]string{
		"Authorization": "Basic ZmFrZUBnbWFpbC5jb206dGVzdGFwaQ==",
	}
	_, bodyStr, err := util.HttpGetResponseContentWithHeaders(urlStr, headers)
	if err != nil {
		return result, err
	}
	if err = json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return result, err
	}

	for _, record := range result {
		record.MDEntryTime = time.Unix(0, record.MDEntryTimeInMS*int64(time.Millisecond))
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
