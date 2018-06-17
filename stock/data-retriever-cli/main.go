package main

import (
	"fmt"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
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
		result, err := aastocks.Quote("00700")
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := bloomberg.Quote("700:HK")
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		result, err := hkex.Quote("700")
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())
	}

	{
		info, err := investtab.GetInfo("00700:HK")
		if err != nil {
			panic(err)
		}
		fmt.Println(info.ToJSONString())

		result, err := investtab.Quote("00700:HK")
		if err != nil {
			panic(err)
		}
		fmt.Println(result.ToJSONString())

		funds, err := investtab.GetFundamentals("00700:HK")
		if err != nil {
			panic(err)
		}
		fmt.Println(funds.ToJSONString())

		didRecords, err := investtab.GetDividendRecords("00700:HK")
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
		pxRecords, err := investtab.GetPriceRecords("00001", beginTime, endTime)
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
		pxRecords, err := yahoo.GetPriceRecords("0001.HK", beginTime, endTime)
		if err != nil {
			panic(err)
		}
		fmt.Println(util.ObjectToJSONString(pxRecords))
	}

}
