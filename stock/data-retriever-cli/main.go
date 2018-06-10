package main

import (
	"fmt"

	"../data-retriever/aastocks"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	result, err := aastocks.Quote("00700")
	if err != nil {
		panic(err)
	}
	fmt.Println(result.ToJSONString())
}
