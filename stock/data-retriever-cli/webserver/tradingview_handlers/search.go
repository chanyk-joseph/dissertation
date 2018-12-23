package tradingview_handlers

import (
	"strconv"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/database"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/common"
	"github.com/labstack/echo"
)

func SearchHandler(c echo.Context) error {
	queryStr := c.FormValue("query")
	//typeStr := c.FormValue("type")
	//exchangeStr := c.FormValue("exchange")
	limit, _ := strconv.Atoi(c.FormValue("limit"))
	if limit == 0 {
		limit = 5
	}

	result, err := database.Query("select DISTINCT(asset_name) from ohlc;")
	if err != nil {
		return c.JSON(500, common.ErrorWrapper{err.Error()})
	}

	type tmpObj struct {
		Symbol      string `json:"symbol"`
		FullName    string `json:"full_name"`
		Description string `json:"description"`
		Exchange    string `json:"exchange"`
		Ticker      string `json:"ticker"`
		Type        string `json:"type"`
	}

	resp := []tmpObj{}
	for _, obj := range result {
		var o tmpObj
		o = tmpObj{
			Symbol:      obj["asset_name"],
			FullName:    obj["asset_name"],
			Description: obj["asset_name"],
			Exchange:    "HSI",
			Ticker:      obj["asset_name"],
			Type:        "stock",
		}
		if strings.Contains(obj["asset_name"], queryStr) && len(resp) < limit {
			resp = append(resp, o)
		}
	}

	return c.JSON(200, resp)
}
