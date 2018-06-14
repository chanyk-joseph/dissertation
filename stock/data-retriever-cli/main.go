package main

import (
	"fmt"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
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
	}
}
