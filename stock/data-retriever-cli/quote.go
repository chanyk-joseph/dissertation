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
	gocache "github.com/patrickmn/go-cache"
)

var cache *gocache.Cache

func init() {
	cache = gocache.New(2*time.Minute, 30*time.Second)
}

func GetQuoteFromAllProviders(symbol models.StandardSymbol) models.QuoteFromAllProviders {
	result := models.QuoteFromAllProviders{}
	result.StandardSymbol = symbol
	result.Quotes = make(map[string]models.StandardQuote)

	mLock := sync.RWMutex{}
	setQuote := func(providerId string, quote models.StandardQuote) {
		mLock.Lock()
		result.Quotes[providerId] = quote
		mLock.Unlock()
	}

	var quoteWaitGroup sync.WaitGroup
	quoteWaitGroup.Add(1)
	go func() {
		tmp := models.StandardQuote{}
		key := "hkex_" + symbol.Symbol
		if entry, found := cache.Get(key); found {
			tmp = entry.(models.StandardQuote)
			setQuote("hkex", tmp)
			quoteWaitGroup.Done()
			return
		}

		if q, err := hkex.Quote(symbol); err == nil {
			tmp.Open = utils.StringToFloat64(q.Open)
			tmp.Low = utils.StringToFloat64(q.Low)
			tmp.High = utils.StringToFloat64(q.High)
			tmp.Close = utils.StringToFloat64(q.LastTradedPrice)
			tmp.Volume = int64(utils.ConvertNumberWithUnitToActualNumber(q.Volume + q.VolumeUnit))
			result.Quotes["hkex"] = tmp
			cache.Set(key, tmp, gocache.DefaultExpiration)
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		tmp := models.StandardQuote{}
		key := "aastocks_" + symbol.Symbol
		if entry, found := cache.Get(key); found {
			tmp = entry.(models.StandardQuote)
			setQuote("aastocks", tmp)
			quoteWaitGroup.Done()
			return
		}

		if q, err := aastocks.Quote(symbol); err == nil {
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.LastTradedPrice
			tmp.Volume = int64(utils.ConvertNumberWithUnitToActualNumber(q.Volume))
			result.Quotes["aastocks"] = tmp
			cache.Set(key, tmp, gocache.DefaultExpiration)
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		tmp := models.StandardQuote{}
		key := "bloomberg_" + symbol.Symbol
		if entry, found := cache.Get(key); found {
			tmp = entry.(models.StandardQuote)
			setQuote("bloomberg", tmp)
			quoteWaitGroup.Done()
			return
		}

		if q, err := bloomberg.Quote(symbol); err == nil {
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.LastTradedPrice
			tmp.Volume = int64(q.Volume)
			result.Quotes["bloomberg"] = tmp
			cache.Set(key, tmp, gocache.DefaultExpiration)
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Add(1)
	go func() {
		tmp := models.StandardQuote{}
		key := "investtab_" + symbol.Symbol
		if entry, found := cache.Get(key); found {
			tmp = entry.(models.StandardQuote)
			setQuote("investtab", tmp)
			quoteWaitGroup.Done()
			return
		}

		if q, err := investtab.Quote(symbol); err == nil {
			tmp.Open = q.Open
			tmp.Low = q.Low
			tmp.High = q.High
			tmp.Close = q.Close
			tmp.Volume = int64(q.Volume)
			result.Quotes["investtab"] = tmp
			cache.Set(key, tmp, gocache.DefaultExpiration)
		}
		quoteWaitGroup.Done()
	}()

	quoteWaitGroup.Wait()
	return result
}

func GetQuotesOfHSIComponents() (result models.QuotesOfHSIComponents, err error) {
	result = models.QuotesOfHSIComponents{}
	hsiComponentsSymbols, err := aastocks.GetHSIConstituentsCodes()
	if err != nil {
		return result, err
	}

	for _, sym := range hsiComponentsSymbols {
		fmt.Println("Quoting " + sym.Symbol)
		q := GetQuoteFromAllProviders(sym)

		if len(q.Quotes) == 0 {
			return result, errors.New("Unable to quote: " + sym.Symbol)
		}
		result.Quotes = append(result.Quotes, q)
	}

	return result, nil
}
