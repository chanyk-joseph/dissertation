package handlers

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/common"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"
	"github.com/labstack/echo"
	"github.com/vjeantet/jodaTime"
)

func SymbolHistoryHandler(c echo.Context) error {
	var err error
	symbolStr := c.Param("symbol")
	dataProvider := c.FormValue("provider")
	resolutionStr := c.FormValue("resolution")
	startTimeStr := c.FormValue("starttime")
	endTimeStr := c.FormValue("endtime")

	requestDataInterval := resolutionStringToIntervalDuration(resolutionStr)
	startTime := 946684800 // 01 Jan 2000 00:00:00
	endTime := int(time.Now().Unix())
	if startTimeStr != "" {
		if startTime, err = strconv.Atoi(startTimeStr); err != nil {
			return c.JSON(500, common.ErrorWrapper{err.Error()})
		}
	}
	if endTimeStr != "" {
		if endTime, err = strconv.Atoi(endTimeStr); err != nil {
			return c.JSON(500, common.ErrorWrapper{err.Error()})
		}
	}
	startTimeUTC := time.Unix(int64(startTime), 0).UTC()
	endTimeUTC := time.Unix(int64(endTime), 0).UTC()

	if dataProvider == "" || dataProvider == "yahoo" {
		stockSymbol, err := utils.ConvertToStandardSymbol(symbolStr)
		if err != nil {
			return c.JSON(500, common.ErrorWrapper{err.Error()})
		}

		var dayResult []yahoo.PriceRecord
		dayResult, err = yahoo.GetPriceRecords(utils.NewStandardSymbol(stockSymbol), startTimeUTC, endTimeUTC)
		if err != nil {
			return c.JSON(500, common.ErrorWrapper{err.Error()})
		}
		return c.JSON(200, genFakeData(dayResult, requestDataInterval))
	}

	if _, err := os.Stat(dataProvider); err != nil {
		return c.JSON(500, common.ErrorWrapper{"Unkonw Data Provider: " + dataProvider})
	}
	targetFolder, _ := filepath.Abs(filepath.Join(dataProvider, symbolStr))
	if CLIUtils.HasFolder(targetFolder) {
		files, _ := CLIUtils.ListFilesWithExtention(targetFolder, ".csv")
		if len(files) == 0 {
			return c.JSON(404, common.ErrorWrapper{"CSV Files Not Found"})
		}
		csvFiles, _ := getCSVFiles(targetFolder, startTimeUTC, endTimeUTC)
		headers, lines, err := loadAndMergeCSVs(csvFiles, startTimeUTC, endTimeUTC)
		if err != nil {
			return c.JSON(500, common.ErrorWrapper{err.Error()})
		}
		return c.JSON(200, csvToObject(headers, lines))
	}
	return c.JSON(500, common.ErrorWrapper{fmt.Sprintf("Symbol(%s) Not Available For Provider(%s)", symbolStr, dataProvider)})
}

func csvToObject(headers []string, lines [][]string) (result []map[string]interface{}) {
	for _, line := range lines {
		obj := map[string]interface{}{}
		for i, cell := range line {
			obj[headers[i]] = cell
		}
		result = append(result, obj)
	}

	return result
}

func guessTimeFormat(str string) (string, error) {
	supporterTimeStr := []string{"YYYY-MM-dd HH:mm:ss:SSSSSS", "YYYY-MM-ddTHH:mm:ss", "YYYY-MM-dd HH:mm:ss", "dd-MM-YYYY HH:mm:ss", "YYYY/MM/dd", "YYYY-MM-dd", "YYYY.MM.dd", "dd/MM/YYYY", "dd-MM-YYYY", "dd.MM.YYYY"}
	targetTimeFormat := ""
	for _, timeFormatStr := range supporterTimeStr {
		if _, err := jodaTime.Parse(timeFormatStr, str); err == nil {
			targetTimeFormat = timeFormatStr
		}
	}
	if targetTimeFormat == "" {
		return targetTimeFormat, errors.New("Unsupported Time Format: " + str)
	}
	return targetTimeFormat, nil
}

func loadAndMergeCSVs(files []string, startTime time.Time, endTime time.Time) (headers []string, lines [][]string, err error) {
	for _, csvFile := range files {
		f, e := os.Open(csvFile)
		if e != nil {
			return headers, lines, e
		}

		var targetTimeFormat string
		// Create a new reader.
		r := csv.NewReader(bufio.NewReader(f))
		for j := 0; true; j++ {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if j == 0 && len(record) > 0 {
				headers = record
			} else {
				if j == 1 {
					targetTimeFormat, err = guessTimeFormat(record[0])
					if err != nil {
						return headers, lines, err
					}
				}

				//Ignore row if the number of column != num of headers
				if len(record) == len(headers) {
					t, err := jodaTime.ParseInLocation(targetTimeFormat, record[0], "Local")
					if err != nil {
						return headers, lines, err
					}
					if t.After(startTime.Add(time.Duration(-1)*time.Second)) && t.Before(endTime.Add(time.Second)) {
						lines = append(lines, record)
					}
				}
			}
		}
	}

	return headers, lines, nil
}

func getCSVFiles(folderPath string, startTime time.Time, endTime time.Time) (result []string, err error) {
	files, _ := CLIUtils.ListFilesWithExtention(folderPath, ".csv")
	if len(files) == 0 {
		return result, nil
	}

	getFileName := func(path string) string {
		return (filepath.Base(path))[:strings.LastIndexByte(filepath.Base(path), '.')]
	}

	firstFileName := getFileName(files[0])
	targetTimeFormat, err := guessTimeFormat(firstFileName)
	if err != nil {
		return result, err
	}

	type tmpStruct struct {
		time time.Time
		path string
	}
	var tmpArray []tmpStruct
	for _, fp := range files {
		o := tmpStruct{}
		o.path = fp
		var err error
		if o.time, err = jodaTime.ParseInLocation(targetTimeFormat, getFileName(fp), "Local"); err == nil {
			tmpArray = append(tmpArray, o)
		}
	}
	sort.Slice(tmpArray, func(i, j int) bool { return tmpArray[i].time.Before(tmpArray[j].time) })
	for _, o := range tmpArray {
		if o.time.After(startTime.Add(time.Duration(-24)*time.Hour)) && o.time.Before(endTime.Add(time.Duration(24)*time.Hour)) {
			result = append(result, o.path)
		}
	}

	return result, nil
}

func genFakeData(dayData []yahoo.PriceRecord, interval time.Duration) (result []yahoo.PriceRecord) {
	totalRecordPerDay := time.Hour * time.Duration(24) / interval
	for _, p := range dayData {
		for j := 0; j < int(totalRecordPerDay); j++ {
			record := yahoo.PriceRecord{}
			record.Date = p.Date.Add(time.Duration(j) * interval)
			record.Open = p.Open
			record.Low = p.Low
			record.High = p.High
			record.Close = p.Close
			record.AdjustedClose = p.AdjustedClose
			result = append(result, record)
		}
	}

	return result
}

func resolutionStringToIntervalDuration(resolutionStr string) time.Duration {
	if resolutionStr == "day" {
		return time.Duration(24) * time.Hour
	} else if resolutionStr == "4hours" {
		return time.Duration(4) * time.Hour
	} else if resolutionStr == "hour" {
		return time.Hour
	} else if resolutionStr == "30minutes" {
		return time.Duration(30) * time.Minute
	} else if resolutionStr == "15minutes" {
		return time.Duration(15) * time.Minute
	} else if resolutionStr == "5minutes" {
		return time.Duration(5) * time.Minute
	} else if resolutionStr == "minute" {
		return time.Minute
	}
	return time.Duration(24) * time.Hour
}
