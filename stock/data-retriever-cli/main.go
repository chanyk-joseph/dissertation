package main

import (
	"fmt"

	"../data-retriever/hkex"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	quote, err := hkex.Quote("700")
	if err != nil {
		panic(err)
	}
	fmt.Println(quote.ToString())
	return

	// var err error
	// var stocks []hkex.Stock

	// _, err = hkex.GetStockList()
	// if err != nil {
	// 	fmt.Println(err)
	// 	return
	// }

	// for _, _ := range stocks {
	// 	fmt.Println(stock.ShortCompanyName + " | " + stock.Symbol)
	// }
	// return
}
