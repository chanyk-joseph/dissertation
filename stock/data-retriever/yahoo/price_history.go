package yahoo

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

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

			toFloat := func(str string) float64 {
				tmp, err := strconv.ParseFloat(str, 64)
				if err != nil {
					panic(err)
				}
				return tmp
			}

			var err error
			priceRecord := PriceRecord{}
			if priceRecord.Date, err = time.Parse(time.RFC3339, rowData[0]+"T00:00:00.000Z"); err != nil {
				panic(err)
			}
			priceRecord.Open = toFloat(rowData[1])
			priceRecord.High = toFloat(rowData[2])
			priceRecord.Low = toFloat(rowData[3])
			priceRecord.Close = toFloat(rowData[4])
			priceRecord.AdjustedClose = toFloat(rowData[5])
			if priceRecord.Volume, err = strconv.ParseInt(rowData[6], 10, 64); err != nil {
				panic(err)
			}

			result = append(result, priceRecord)
		}
	})

	urlStr := fmt.Sprintf("https://finance.yahoo.com/quote/%s/history?period1=%d&period2=%d&interval=1d&filter=history&frequency=1d", symbol, startTime.Unix(), endTime.Unix())
	fmt.Println(urlStr)
	c.Visit(urlStr)

	if len(result) == 0 {
		return result, errors.New("Unable To Get History Stock Price Of " + symbol)
	}

	return result, nil
}
