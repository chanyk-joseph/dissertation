package main

import (
	"fmt"
	"time"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
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
		_, q, _ := CLIUtils.GetQuoteFromAllProviders(utils.NewStandardSymbol("00001.HK"))
		fmt.Println(utils.ObjectToJSONString(q))

		r, _, _ := CLIUtils.GetQuoteFromAllProviders(utils.NewStandardSymbol("00762.HK"))
		fmt.Println(utils.ObjectToJSONString(r))

		hsiQ, _ := CLIUtils.GetQuotesOfHSIComponents()
		fmt.Println(utils.ObjectToJSONString(hsiQ))
	}

	{
		startTimeUTC := time.Unix(int64(1530457035), 0).UTC()
		endTimeUTC := time.Unix(int64(1530716235), 0).UTC()
		var dayResult []yahoo.PriceRecord
		var err error
		dayResult, err = yahoo.GetPriceRecords(utils.NewStandardSymbol("700"), startTimeUTC, endTimeUTC)
		if err != nil {
			panic(err)
		}
		fmt.Println(utils.ObjectToJSONString(dayResult))

		timeToEpochSeconds := func(input interface{}) interface{} {
			return input.(time.Time).Unix()
		}
		customFieldNames := map[string]string{"Date": "t", "Open": "o", "High": "h", "Low": "l", "AdjustedClose": "c", "Volume": "v"}
		customTranslateFuncs := map[string]CLIUtils.TranslateFunction{"Date": timeToEpochSeconds}
		result := CLIUtils.ArrToUDF(dayResult, customFieldNames, customTranslateFuncs)
		fmt.Println(utils.ObjectToJSONString(result))
	}

	fmt.Println("Press the Enter Key To Stop")
	var input string
	fmt.Scanln(&input)
}
