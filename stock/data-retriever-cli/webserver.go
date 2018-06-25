package main

import (
	"errors"
	"strconv"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
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
	e.GET("/history/:symbol", func(c echo.Context) error {
		stockSymbol, err := utils.ConvertToStandardSymbol(c.Param("symbol"))
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		}

		resolutionStr := c.FormValue("resolution")
		startTimeStr := c.FormValue("starttime")
		endTimeStr := c.FormValue("endtime")

		startTime := 946684800 // 01 Jan 2000 00:00:00
		endTime := int(time.Now().Unix())
		if startTimeStr != "" {
			if startTime, err = strconv.Atoi(startTimeStr); err != nil {
				return c.JSON(500, ErrorWrapper{err.Error()})
			}
		}
		if endTimeStr != "" {
			if startTime, err = strconv.Atoi(startTimeStr); err != nil {
				return c.JSON(500, ErrorWrapper{err.Error()})
			}
		}

		var dayResult []yahoo.PriceRecord
		dayResult, err = yahoo.GetPriceRecords(utils.NewStandardSymbol(stockSymbol), time.Unix(int64(startTime), 0), time.Unix(int64(endTime), 0))
		if err != nil {
			return c.JSON(500, ErrorWrapper{err.Error()})
		} else if resolutionStr == "" || resolutionStr == "day" {
			return c.JSON(200, dayResult)
		} else if resolutionStr == "hour" {
			var hourResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for h := 0; h < 24; h++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(h) * time.Hour)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					hourResult = append(hourResult, record)
				}
			}
			return c.JSON(200, hourResult)
		} else if resolutionStr == "4hours" {
			var fourHoursResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for h := 0; h < 6; h++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(4*h) * time.Hour)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					fourHoursResult = append(fourHoursResult, record)
				}
			}
			return c.JSON(200, fourHoursResult)
		} else if resolutionStr == "minute" {
			var minuteResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for m := 0; m < 24*60; m++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(m) * time.Minute)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					minuteResult = append(minuteResult, record)
				}
			}
			return c.JSON(200, minuteResult)
		} else if resolutionStr == "30minutes" {
			var thirtyMinutesResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for m := 0; m < 24*2; m++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(30*m) * time.Minute)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					thirtyMinutesResult = append(thirtyMinutesResult, record)
				}
			}
			return c.JSON(200, thirtyMinutesResult)
		} else if resolutionStr == "15minutes" {
			var fifteenMinutesResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for m := 0; m < 24*4; m++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(15*m) * time.Minute)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					fifteenMinutesResult = append(fifteenMinutesResult, record)
				}
			}
			return c.JSON(200, fifteenMinutesResult)
		} else if resolutionStr == "5minutes" {
			var fivenMinutesResult []yahoo.PriceRecord
			for _, p := range dayResult {
				for m := 0; m < 24*12; m++ {
					record := yahoo.PriceRecord{}
					record.Date = p.Date.Add(time.Duration(5*m) * time.Minute)
					record.Open = p.Open
					record.Low = p.Low
					record.High = p.High
					record.Close = p.Close
					record.AdjustedClose = p.AdjustedClose
					fivenMinutesResult = append(fivenMinutesResult, record)
				}
			}
			return c.JSON(200, fivenMinutesResult)
		}
		return c.JSON(500, ErrorWrapper{errors.New("Unkown resolution: " + resolutionStr).Error()})
	})

	return e
}
