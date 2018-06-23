package main

import (
	"sync"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
)

func GetQuoteFromAllProviders(symbol models.StandardSymbol) models.QuoteFromAllProviders {
	result := models.QuoteFromAllProviders{}
	result.StandardSymbol = symbol
	result.Quotes = make(map[string]models.StandardQuote)

	var quoteWaitGroup sync.WaitGroup
	quoteWaitGroup.Add(1)
	go func() {
		if q, err := hkex.Quote(symbol); err == nil {
			tmp := models.StandardQuote{}
			tmp.Open = util.StringToFloat64(q.Open)
			tmp.Low = util.StringToFloat64(q.Low)
			tmp.High = util.StringToFloat64(q.High)
			tmp.Close = util.StringToFloat64(q.LastTradedPrice)
			tmp.Volume = int64(util.ConvertNumberWithUnitToActualNumber(q.Volume + q.VolumeUnit))
			result.Quotes["hkex"] = tmp
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		if q, err := aastocks.Quote(symbol); err == nil {
			tmp := models.StandardQuote{}
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.LastTradedPrice
			tmp.Volume = int64(util.ConvertNumberWithUnitToActualNumber(q.Volume))
			result.Quotes["aastocks"] = tmp
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		if q, err := bloomberg.Quote(symbol); err == nil {
			tmp := models.StandardQuote{}
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.LastTradedPrice
			tmp.Volume = int64(q.Volume)
			result.Quotes["bloomberg"] = tmp
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		if q, err := investtab.Quote(symbol); err == nil {
			tmp := models.StandardQuote{}
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.Close
			tmp.Volume = int64(q.Volume)
			result.Quotes["investtab"] = tmp
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Wait()
	return result
}
