package main

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"

	"github.com/karlseguin/ccache"
)

var cache *ccache.Cache

func init() {
	cache = ccache.New(ccache.Configure())
}

func GetQuoteFromAllProviders(symbol models.StandardSymbol) (models.QuoteFromAllProviders, error) {
	item, err := cache.Fetch(symbol.Symbol, 2*time.Minute, func() (interface{}, error) {
		result := models.QuoteFromAllProviders{}
		result.StandardSymbol = symbol
		result.Quotes = make(map[string]models.StandardQuote)

		mLock := make(chan bool, 1)
		setQuote := func(providerId string, quote models.StandardQuote) {
			mLock <- true
			result.Quotes[providerId] = quote
			<-mLock
		}

		var quoteWaitGroup sync.WaitGroup
		quoteWaitGroup.Add(1)
		go func() {
			tmp := models.StandardQuote{}
			q, err := hkex.Quote(symbol)
			if err == nil {
				tmp.Open = utils.StringToFloat64(q.Open)
				tmp.Low = utils.StringToFloat64(q.Low)
				tmp.High = utils.StringToFloat64(q.High)
				tmp.Close = utils.StringToFloat64(q.LastTradedPrice)
				tmp.Volume = int64(utils.ConvertNumberWithUnitToActualNumber(q.Volume + q.VolumeUnit))
				setQuote("hkex", tmp)
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			tmp := models.StandardQuote{}
			q, err := aastocks.Quote(symbol)
			if err == nil {
				tmp.Open = q.Open
				tmp.Low = q.Low
				tmp.High = q.High
				tmp.Close = q.LastTradedPrice
				tmp.Volume = int64(utils.ConvertNumberWithUnitToActualNumber(q.Volume))
				setQuote("aastocks", tmp)
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			tmp := models.StandardQuote{}
			q, err := bloomberg.Quote(symbol)
			if err == nil {
				tmp.Open = q.Open
				tmp.Low = q.Low
				tmp.High = q.High
				tmp.Close = q.LastTradedPrice
				tmp.Volume = int64(q.Volume)
				setQuote("bloomberg", tmp)
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			tmp := models.StandardQuote{}
			q, err := investtab.Quote(symbol)
			if err == nil {
				tmp.Open = q.Open
				tmp.Low = q.Low
				tmp.High = q.High
				tmp.Close = q.Close
				tmp.Volume = int64(q.Volume)
				setQuote("investtab", tmp)
			}
			quoteWaitGroup.Done()
		}()

		quoteWaitGroup.Wait()
		if len(result.Quotes) == 0 {
			return nil, errors.New("No Quote From All Providers")
		}
		return result, nil
	})

	if err != nil {
		return models.QuoteFromAllProviders{}, err
	}
	return item.Value().(models.QuoteFromAllProviders), nil
}

func GetQuotesOfHSIComponents() (result models.QuotesOfHSIComponents, err error) {
	result = models.QuotesOfHSIComponents{}
	hsiComponentsSymbols, err := aastocks.GetHSIConstituentsCodes()
	if err != nil {
		return result, err
	}

	for _, sym := range hsiComponentsSymbols {
		fmt.Println("Quoting " + sym.Symbol)
		q, err := GetQuoteFromAllProviders(sym)

		if err != nil {
			return result, err
		}
		result.Quotes = append(result.Quotes, q)
	}

	return result, nil
}
