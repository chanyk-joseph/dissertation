package main

import (
	"fmt"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// Examples of API
func Examples() {
	{
		result, err := aastocks.Quote(utils.NewStandardSymbol("00700"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := bloomberg.Quote(utils.NewStandardSymbol("700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := hkex.Quote(utils.NewStandardSymbol("700"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		info, err := investtab.GetInfo(utils.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(info.ToJSONString())

		result, err := investtab.Quote(utils.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())

		funds, err := investtab.GetFundamentals(utils.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(funds.ToJSONString())

		didRecords, err := investtab.GetDividendRecords(utils.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(utils.ObjectToJSONString(didRecords))

		dateString := "1960-11-12T00:00:00.000Z"
		beginTime, err := time.Parse(time.RFC3339, dateString)
		if err != nil {
			panic(err)
		}
		dateString = "2007-11-12T00:00:00.000Z"
		endTime, err := time.Parse(time.RFC3339, dateString)
		if err != nil {
			panic(err)
		}
		pxRecords, err := investtab.GetPriceRecords(utils.NewStandardSymbol("00001"), beginTime, endTime)
		if err != nil {
			panic(err)
		}
		fmt.Println(utils.ObjectToJSONString(pxRecords))
	}

	{
		dateString := "1960-11-12T00:00:00.000Z"
		beginTime, err := time.Parse(time.RFC3339, dateString)
		if err != nil {
			panic(err)
		}
		dateString = "2007-11-12T00:00:00.000Z"
		endTime, err := time.Parse(time.RFC3339, dateString)
		if err != nil {
			panic(err)
		}
		pxRecords, err := yahoo.GetPriceRecords(utils.NewStandardSymbol("1:HK"), beginTime, endTime)
		if err != nil {
			panic(err)
		}
		fmt.Println(utils.ObjectToJSONString(pxRecords))
	}

	str := utils.NewStandardSymbol("12")
	fmt.Println(str.Symbol)

	{
		_, q, _ := GetQuoteFromAllProviders(utils.NewStandardSymbol("00001.HK"))
		fmt.Println(utils.ObjectToJSONString(q))

		r, _, _ := GetQuoteFromAllProviders(utils.NewStandardSymbol("00762.HK"))
		fmt.Println(utils.ObjectToJSONString(r))

		hsiQ, _ := GetQuotesOfHSIComponents()
		fmt.Println(utils.ObjectToJSONString(hsiQ))
	}
}
