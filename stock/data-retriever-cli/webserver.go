package main

import (
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
	"github.com/labstack/echo/middleware"
	"github.com/vjeantet/jodaTime"
)

type ErrorWrapper struct {
	Error string `json:"error"`
}

// SetupWebserver return a Rest API server for stock queries
func SetupWebserver() *echo.Echo {
	e := echo.New()
	e.Use(middleware.CORSWithConfig(middleware.CORSConfig{
		AllowOrigins: []string{"*"},
		AllowMethods: []string{echo.GET, echo.HEAD, echo.PUT, echo.PATCH, echo.POST, echo.DELETE},
	}))
	e.GET("/", func(c echo.Context) error {
		return c.HTML(200, string("Joseph Stock Server"))
	})
	e.GET("/hsicomponents", func(c echo.Context) error {
		stockSymbols, err := aastocks.GetHSIConstituentsCodes()
		if err == nil {
			result := []string{}
			for _, obj := range stockSymbols {
				result = append(result, obj.Symbol)
			}
			return c.JSON(200, result)
		}
		return c.JSON(500, ErrorWrapper{err.Error()})
	})
	e.GET("/hsicomponents/quote", func(c echo.Context) error {
		q, err := GetQuotesOfHSIComponents()
		if err == nil {
			return c.JSON(200, q)
		}
		return c.JSON(500, ErrorWrapper{err.Error()})
	})
	e.GET("/quote/:symbol", func(c echo.Context) error {
		stockSymbol, err := utils.ConvertToStandardSymbol(c.Param("symbol"))
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
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

		rawQuote, q, err := GetQuoteFromAllProviders(utils.NewStandardSymbol(stockSymbol))
		allQuote.Symbol = q.Symbol
		allQuote.Standard = q.Quotes
		allQuote.Raw = rawQuote.Quotes
		if err == nil {
			if bShowRaw {
				return c.JSON(200, allQuote)
			}
			return c.JSON(200, q)
		}
		return c.JSON(404, ErrorWrapper{err.Error()})
	})
	e.GET("/daily_history/:symbol", func(c echo.Context) error {
		stockSymbol, err := utils.ConvertToStandardSymbol(c.Param("symbol"))
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		}

		t1 := c.FormValue("startdate")
		t2 := c.FormValue("enddate")
		if t1 == "" {
			t1 = "2000-01-01"
		}
		if t2 == "" {
			t2 = jodaTime.Format("YYYY-MM-dd", time.Now().UTC())
		}

		startTime, err := jodaTime.Parse("YYYY-MM-dd", t1)
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		}
		endTime, err := jodaTime.Parse("YYYY-MM-dd", t2)
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		}

		result, err := yahoo.GetPriceRecords(utils.NewStandardSymbol(stockSymbol), startTime, endTime)
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		}
		return c.JSON(200, result)
	})

	return e
}
