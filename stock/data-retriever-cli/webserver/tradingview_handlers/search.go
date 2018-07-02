package tradingview_handlers

import (
	"strconv"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/common"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
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

	stockSymbols, err := aastocks.GetHSIConstituentsCodes()
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
	for _, obj := range stockSymbols {
		var o tmpObj
		o = tmpObj{
			Symbol:      obj.Symbol,
			FullName:    obj.Symbol,
			Description: obj.Symbol,
			Exchange:    "HSI",
			Ticker:      obj.Symbol,
			Type:        "stock",
		}
		if strings.Contains(obj.Symbol, queryStr) && len(resp) < limit {
			resp = append(resp, o)
		}
	}

	return c.JSON(200, resp)
}
