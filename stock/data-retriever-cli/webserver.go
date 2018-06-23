package main

import (
	"errors"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
	"github.com/labstack/echo/middleware"
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
		stockSymbol := c.Param("symbol")
		q := GetQuoteFromAllProviders(utils.NewStandardSymbol(stockSymbol))
		if len(q.Quotes) > 0 {
			return c.JSON(200, q)
		}
		err := errors.New("No Quote Result From All Data Providers")
		return c.JSON(404, ErrorWrapper{err.Error()})
	})

	return e
}
