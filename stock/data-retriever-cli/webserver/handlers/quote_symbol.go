package handlers

import (
	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/common"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
)

func QuoteSymbolHandler(c echo.Context) error {
	stockSymbol, err := utils.ConvertToStandardSymbol(c.Param("symbol"))
	if err != nil {
		return c.JSON(500, common.ErrorWrapper{err.Error()})
	}
	bShowRaw := false
	tmp := c.FormValue("raw")
	if tmp == "true" {
		bShowRaw = true
	}

	allQuote := struct {
		models.StandardSymbol
		Standard map[string]models.StandardQuote `json:"quotes"`
		Raw      map[string]interface{}          `json:"raw"`
	}{}

	rawQuote, q, err := CLIUtils.GetQuoteFromAllProviders(utils.NewStandardSymbol(stockSymbol))
	allQuote.Symbol = q.Symbol
	allQuote.Standard = q.Quotes
	allQuote.Raw = rawQuote.Quotes
	if err == nil {
		if bShowRaw {
			return c.JSON(200, allQuote)
		}
		return c.JSON(200, q)
	}
	return c.JSON(404, common.ErrorWrapper{err.Error()})
}
