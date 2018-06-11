package main

import (
	"fmt"

	"../data-retriever/aastocks"
	"../data-retriever/bloomberg"
	"../data-retriever/hkex"
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
}
