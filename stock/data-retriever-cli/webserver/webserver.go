package webserver

import (
	"strconv"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/embeded_tradingview_html"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/handlers"
	tv "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/tradingview_handlers"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
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
	}), middleware.GzipWithConfig(middleware.GzipConfig{
		Level: 1,
	}))
	e.GET("/", func(c echo.Context) error {
		return c.HTML(200, string("Joseph Stock API Server"))
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
		q, err := CLIUtils.GetQuotesOfHSIComponents()
		if err == nil {
			return c.JSON(200, q)
		}
		return c.JSON(500, ErrorWrapper{err.Error()})
	})
	e.GET("/quote/:symbol", handlers.QuoteSymbolHandler)
	e.GET("/history/:symbol", handlers.SymbolHistoryHandler)

	setupTradingviewAPI(e)
	setupTradingviewUI(e)

	return e
}

func setupTradingviewAPI(e *echo.Echo) {
	e.GET("/tradingview-udf-api/config", tv.ConfigHandler)
	e.GET("/tradingview-udf-api/symbols", tv.SymbolHandler)
	e.GET("/tradingview-udf-api/search", tv.SearchHandler)
	e.GET("/tradingview-udf-api/history", tv.HistoryHandler)
	e.GET("/tradingview-udf-api/quotes", tv.QuotesHandler)

	e.GET("/tradingview-udf-api/time", func(c echo.Context) error {
		return c.String(200, strconv.Itoa(int(time.Now().UTC().Unix())))
	})

	e.GET("/tradingview-storage-api/1.1/charts", tv.GetChartHandler)
	e.POST("/tradingview-storage-api/1.1/charts", tv.PostChartHandler)
	e.DELETE("/tradingview-storage-api/1.1/charts", tv.DeleteChartHandler)

	e.GET("/tradingview-storage-api/1.1/study_templates", tv.GetStudyTemplatesHandler)
	e.POST("/tradingview-storage-api/1.1/study_templates", tv.PostStudyTemplatesHandler)
	e.DELETE("/tradingview-storage-api/1.1/study_templates", tv.DeleteStudyTemplatesHandler)

	e.GET("/tradingview-backtest-api/records", tv.GetBacktestAnnotationHandler)

	// TBD
	// e.GET("/tradingview-udf-api/marks", tv.TBD)
	// e.GET("/tradingview-udf-api/timescale_marks", tv.TBD)
}

func setupTradingviewUI(e *echo.Echo) {
	e.GET("/chart/*", echo.WrapHandler(embeded_tradingview_html.Handler))
}
