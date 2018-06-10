package main

import (
	"../data-retriever/hkex"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	_, err := hkex.Quote("700")
	if err != nil {
		panic(err)
	}

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
	return
}
