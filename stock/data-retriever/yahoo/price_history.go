package yahoo

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/gocolly/colly"
	"github.com/oliveagle/jsonpath"
)

type PriceRecord struct {
	Date          time.Time
	Open          float64
	High          float64
	Low           float64
	Close         float64
	AdjustedClose float64
	Volume        int64
}

func GetPriceRecords(symbol string, startTime time.Time, endTime time.Time) ([]PriceRecord, error) {
	result := []PriceRecord{}

	// Symbol Example: 0001.HK

	c := colly.NewCollector()
	detailCollector := c.Clone()

	c.OnHTML("html", func(e *colly.HTMLElement) {
		if e.Response.StatusCode != http.StatusOK {
			return
		}

		bodyStr := string(e.Response.Body)

		var re = regexp.MustCompile(`(?m)^root.App.main = (.*);`)
		match := re.FindStringSubmatch(bodyStr)
		jsonStr := match[1]

		var obj interface{}
		json.Unmarshal([]byte(jsonStr), &obj)
		res, err := jsonpath.JsonPathLookup(obj, "$.context.dispatcher.stores.CrumbStore.crumb")
		if err != nil {
			panic(err)
		}
		accessToken := res.(string)

		urlStr := fmt.Sprintf("https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%d&period2=%d&interval=1d&events=history&crumb=%s", symbol, startTime.Unix(), endTime.Unix(), accessToken)
		fmt.Println(urlStr)
		detailCollector.Visit(urlStr)
	})

	detailCollector.OnResponse(func(res *colly.Response) {
		if res.StatusCode != http.StatusOK {
			return
		}

		csvData := strings.Split(string(res.Body), "\n")
		for i := 1; i < len(csvData); i++ {
			rowData := strings.Split(csvData[i], ",")
			if len(rowData) != 7 || rowData[6] == "null" {
				continue
			}

			var err error
			priceRecord := PriceRecord{}
			if priceRecord.Date, err = time.Parse(time.RFC3339, rowData[0]+"T00:00:00.000Z"); err != nil {
				panic(err)
			}
			priceRecord.Open = util.StringToFloat64(rowData[1])
			priceRecord.High = util.StringToFloat64(rowData[2])
			priceRecord.Low = util.StringToFloat64(rowData[3])
			priceRecord.Close = util.StringToFloat64(rowData[4])
			priceRecord.AdjustedClose = util.StringToFloat64(rowData[5])
			priceRecord.Volume = util.StringToInt64(rowData[6])

			result = append(result, priceRecord)
		}
	})

	urlStr := fmt.Sprintf("https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d", symbol, startTime.Unix(), endTime.Unix())
	fmt.Println(urlStr)
	c.Visit(urlStr)

	if len(result) == 0 {
		return result, errors.New("Unable To Get Stock Price History For " + symbol)
	}

	return result, nil
}
