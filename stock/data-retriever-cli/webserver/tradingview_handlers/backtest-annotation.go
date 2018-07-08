package tradingview_handlers

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/labstack/echo"
)

func GetBacktestAnnotationHandler(c echo.Context) error {
	symbolStr := c.FormValue("symbol")
	if strings.Contains(symbolStr, ":") {
		symbolStr = strings.Split(symbolStr, ":")[1]
	}

	type BackTestRecord struct {
		Timestamp int64   `json:"time"`
		Type      string  `json:"type"`
		Price     float64 `json:"price"`
		Unit      int     `json:"unit"`
	}
	type Records struct {
		Records []BackTestRecord `json:"records"`
	}
	result := Records{}
	result.Records = []BackTestRecord{}

	backtestFile, _ := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), "backtest", symbolStr, "buy_sell_records.json"))
	if !CLIUtils.HasFile(backtestFile) {
		return c.JSON(200, result)
	}
	jsonContent, err := ioutil.ReadFile(backtestFile)
	if err != nil {
		return c.JSON(500, result)
	}
	err = json.Unmarshal(jsonContent, &result)
	if err != nil {
		return c.JSON(500, result)
	}

	return c.JSON(200, result)
}
