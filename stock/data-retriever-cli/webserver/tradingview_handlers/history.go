package tradingview_handlers

import (
	"fmt"
	"strconv"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/database"
	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/labstack/echo"
	"github.com/vjeantet/jodaTime"
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

	if symbolStr == "joseph-indicator" {
		symbolStr = "HKGIDXHKD"
	}

	// Check available resolutions
	result, err := database.Query(fmt.Sprintf("select DISTINCT(resolution) from ohlc WHERE asset_name = '%s';", symbolStr))
	if err != nil {
		str := err.Error()
		return c.JSON(200, errStruct{"error", &str, nil})
	}
	availableResolutions := make(map[string]bool)
	for _, row := range result {
		if tvRepresentation, ok := DBResolutionToTVResolutionMap[row["resolution"]]; ok {
			availableResolutions[tvRepresentation] = true
		}
	}
	if !availableResolutions[resolutionStr] {
		str := fmt.Sprintf("Resolution %s Not Available For Asset %s", resolutionStr, symbolStr)
		return c.JSON(200, errStruct{"error", &str, nil})
	}

	// Get History
	timeFormat := "2006-01-02 15:04:05"
	result, err = database.Query(fmt.Sprintf("select * from ohlc WHERE asset_name = '%s' AND `resolution` = '%s' AND `datetime` >= '%s' AND `datetime` <= '%s';", symbolStr, TVResolutionToDBResolutionMap[resolutionStr], startTimeUTC.Format(timeFormat), endTimeUTC.Format(timeFormat)))
	if err != nil {
		str := err.Error()
		return c.JSON(200, errStruct{"error", &str, nil})
	}

	timeToEpochSeconds := func(input interface{}) interface{} {
		datetimeStr := input.(string)
		t, err := jodaTime.Parse("YYYY-MM-dd HH:mm:ss", datetimeStr)
		if err != nil {
			panic(err)
		}
		return t.Unix()
	}
	stringToFloat64 := func(input interface{}) interface{} {
		valStr := input.(string)
		result, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			panic(err)
		}
		return result
	}

	//AdjustedClose
	customFieldNames := map[string]string{"datetime": "t", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
	customTranslateFuncs := map[string]CLIUtils.TranslateFunction{
		"datetime": timeToEpochSeconds,
		"open":     stringToFloat64,
		"high":     stringToFloat64,
		"low":      stringToFloat64,
		"close":    stringToFloat64,
		"volume":   stringToFloat64,
	}
	out := CLIUtils.ArrToUDF(result, customFieldNames, customTranslateFuncs)

	return c.JSON(200, out)
}

/*
func HistoryHandler_bk(c echo.Context) error {
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

	if strings.Contains(symbolStr, "_backtest") {
		type BackTestRecord struct {
			Timestamp int64  `json:"time"`
			Type      string `json:"type"`
			Price     string `json:"price"`
			Unit      string `json:"unit"`
		}
		type Records struct {
			Records []BackTestRecord `json:"records"`
		}
		tmp := strings.Split(symbolStr, "_")
		backtestFile, _ := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), "backtest", tmp[0], "buy_sell_records.json"))
		if !CLIUtils.HasFile(backtestFile) {
			str := "Backtest Records Not Found"
			return c.JSON(200, errStruct{"error", &str, nil})
		}
		var records Records
		jsonContent, err := ioutil.ReadFile(backtestFile)
		if err != nil {
			str := err.Error()
			return c.JSON(200, errStruct{"error", &str, nil})
		}
		json.Unmarshal(jsonContent, &records)
		// profitAndLost := 0
		// position := 0
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

	timeToEpochSeconds := func(input interface{}) interface{} {
		return input.(time.Time).Unix()
	}
	//AdjustedClose
	customFieldNames := map[string]string{"Date": "t", "Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"}
	customTranslateFuncs := map[string]CLIUtils.TranslateFunction{"Date": timeToEpochSeconds}
	result := CLIUtils.ArrToUDF(tmpResult, customFieldNames, customTranslateFuncs)

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
*/
