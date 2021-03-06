package yahoo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/gocolly/colly"
	"github.com/oliveagle/jsonpath"
	"github.com/vjeantet/jodaTime"
)

type PriceRecord struct {
	Date          time.Time `json:"date"`
	Open          float64   `json:"open"`
	High          float64   `json:"high"`
	Low           float64   `json:"low"`
	Close         float64   `json:"close"`
	AdjustedClose float64   `json:"adjusted_close"`
	Volume        int64     `json:"volume"`
}

func (p *PriceRecord) MarshalJSON() ([]byte, error) {
	type Alias PriceRecord
	return json.Marshal(&struct {
		*Alias
		Date string `json:"date"`
	}{
		Alias: (*Alias)(p),
		Date:  jodaTime.Format("YYYY-MM-ddTHH:mm:ss.000Z", p.Date),
	})
}

func GetPriceRecords(standardSymbol models.StandardSymbol, startTime time.Time, endTime time.Time) ([]PriceRecord, error) {
	result := []PriceRecord{}
	symbol := standardSymbol.Symbol[1:]
	if standardSymbol.Symbol == "HSI" {
		symbol = "%5EHSI"
	}

	c := colly.NewCollector()
	detailCollector := c.Clone()
	var parseError error
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
			parseError = err
			return
		}
		accessToken := res.(string)

		urlStr := fmt.Sprintf("https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%d&period2=%d&interval=1d&events=history&crumb=%s", symbol, startTime.Unix(), endTime.Unix(), accessToken)
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
				parseError = err
				return
			}
			priceRecord.Open = utils.StringToFloat64(rowData[1])
			priceRecord.High = utils.StringToFloat64(rowData[2])
			priceRecord.Low = utils.StringToFloat64(rowData[3])
			priceRecord.Close = utils.StringToFloat64(rowData[4])
			priceRecord.AdjustedClose = utils.StringToFloat64(rowData[5])
			priceRecord.Volume = utils.StringToInt64(rowData[6])

			result = append(result, priceRecord)
		}
	})

	urlStr := fmt.Sprintf("https://finance.yahoo.com/quote/%s/history", symbol)
	c.Visit(urlStr)

	if parseError != nil {
		return result, fmt.Errorf("Unable To Get Stock Price History For %s: \n%s", symbol, parseError.Error())
	}
	if len(result) == 0 {
		return result, fmt.Errorf("Unable To Get Stock Price History For %s", symbol)
	}

	return result, nil
}
