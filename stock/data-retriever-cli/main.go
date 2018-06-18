package main

import (
	"fmt"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/converter"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/yahoo"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	{
		result, err := aastocks.Quote(converter.NewStandardSymbol("00700"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := bloomberg.Quote(converter.NewStandardSymbol("700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := hkex.Quote(converter.NewStandardSymbol("700"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		info, err := investtab.GetInfo(converter.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(info.ToJSONString())

		result, err := investtab.Quote(converter.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())

		funds, err := investtab.GetFundamentals(converter.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(funds.ToJSONString())

		didRecords, err := investtab.GetDividendRecords(converter.NewStandardSymbol("00700:HK"))
		if err != nil {
			panic(err)
		}
		fmt.Println(util.ObjectToJSONString(didRecords))

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
		pxRecords, err := investtab.GetPriceRecords(converter.NewStandardSymbol("00001"), beginTime, endTime)
		if err != nil {
			panic(err)
		}
		fmt.Println(util.ObjectToJSONString(pxRecords))
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
		pxRecords, err := yahoo.GetPriceRecords(converter.NewStandardSymbol("1:HK"), beginTime, endTime)
		if err != nil {
			panic(err)
		}
		fmt.Println(util.ObjectToJSONString(pxRecords))
	}

	str := converter.NewStandardSymbol("12")
	fmt.Println(str.Symbol)
}
