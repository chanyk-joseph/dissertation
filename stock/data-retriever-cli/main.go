package main

import (
	"fmt"

	"../data-retriever/bloomberg"
	"../data-retriever/common/util"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	result, err := bloomberg.Quote("700:HK")
	if err != nil {
		panic(err)
	}
	fmt.Println(util.ObjectToJsonString(result))
}
