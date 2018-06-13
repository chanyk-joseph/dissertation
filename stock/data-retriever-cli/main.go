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
		investtab.GetInfo("00001:HK")
		investtab.Test()
	}
}
