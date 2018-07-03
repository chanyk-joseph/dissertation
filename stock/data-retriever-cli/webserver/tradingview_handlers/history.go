package tradingview_handlers

import (
	"strconv"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"
	"github.com/labstack/echo"
)

func HistoryHandler(c echo.Context) error {
	symbolStr := c.FormValue("symbol")
	fromTime, _ := strconv.Atoi(c.FormValue("from"))
	toTime, _ := strconv.Atoi(c.FormValue("to"))
	resolutionStr := c.FormValue("resolution")

	startTimeUTC := time.Unix(int64(fromTime), 0).UTC()
	endTimeUTC := time.Unix(int64(toTime), 0).UTC()

	type errStruct struct {
		Status   string  `json:"s"`
		ErrorMsg *string `json:"errmsg"`
		NextTime *int64  `json:"nextTime"`
	}

	var dayResult []yahoo.PriceRecord
	var err error
	dayResult, err = yahoo.GetPriceRecords(utils.NewStandardSymbol(symbolStr), startTimeUTC, endTimeUTC)
	if err != nil {
		str := err.Error()
		return c.JSON(200, errStruct{"error", &str, nil})
	}

	tmpResult := genFakeData(dayResult, resolutionStringToIntervalDuration(resolutionStr))
	if len(tmpResult) == 0 {
		t := dayResult[len(dayResult)-1].Date.Unix()
		return c.JSON(200, errStruct{"no_data", nil, &t})
	}

	type tmpObj struct {
		Status  string    `json:"s"`
		BarTime []int64   `json:"t"`
		Close   []float64 `json:"c"`
		Open    []float64 `json:"o"`
		High    []float64 `json:"h"`
		Low     []float64 `json:"l"`
		Volume  []int64   `json:"v"`
	}
	result := tmpObj{Status: "ok"}
	for _, o := range tmpResult {
		result.BarTime = append(result.BarTime, o.Date.Unix())
		result.Close = append(result.Close, o.AdjustedClose)
		result.Open = append(result.Open, o.Open)
		result.High = append(result.High, o.High)
		result.Low = append(result.Low, o.Low)
		result.Volume = append(result.Volume, o.Volume)
	}

	return c.JSON(200, result)
}

func genFakeData(dayData []yahoo.PriceRecord, interval time.Duration) (result []yahoo.PriceRecord) {
	lastTime := time.Unix(315532800, 0).UTC()
	totalRecordPerMonth := time.Hour * time.Duration(24*30) / interval

	if interval == time.Duration(24*30)*time.Hour && len(dayData) > 0 {
		// 1M
		lastMonth := dayData[0].Date.Add(time.Duration(-24*30) * time.Hour).Month()
		for _, p := range dayData {
			if p.Date.Month() != lastMonth {
				result = append(result, p)
				lastMonth = p.Date.Month()
			}
		}
	} else if interval == time.Duration(24)*time.Hour && len(dayData) > 0 {
		// D
		return dayData
	} else {
		for i, p := range dayData {
			if i == 0 {
				lastTime = p.Date.Add(-1 * interval)
			}

			if p.Date.Sub(lastTime).Seconds() >= interval.Seconds() {
				for j := 0; j < int(totalRecordPerMonth); j++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(j) * interval)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.Volume = p.Volume
					record.AdjustedClose = p.AdjustedClose
					result = append(result, record)
				}
				lastTime = lastTime.Add(interval)
			}
		}
	}

	return result
}

func resolutionStringToIntervalDuration(resolutionStr string) time.Duration {
	if resolutionStr == "1M" {
		return time.Duration(24*30) * time.Hour
	} else if resolutionStr == "1W" {
		return time.Duration(24*7) * time.Hour
	} else if resolutionStr == "1D" {
		return time.Duration(24) * time.Hour
	} else if resolutionStr == "240" {
		return time.Duration(4) * time.Hour
	} else if resolutionStr == "60" {
		return time.Hour
	} else if resolutionStr == "30" {
		return time.Duration(30) * time.Minute
	} else if resolutionStr == "15" {
		return time.Duration(15) * time.Minute
	} else if resolutionStr == "5" {
		return time.Duration(5) * time.Minute
	} else if resolutionStr == "1" {
		return time.Minute
	}
	return time.Duration(24) * time.Hour
}
