package main

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	app           = kingpin.New("data-retriever", "Serve stock quote via Web Interface / save to mysql database")
	serverCommand = app.Command("web", "Web Server Mode")
	serverPort    = serverCommand.Flag("port", "Port For Listening Slave Machine, Default = 8888").Default("8888").Short('p').Int()
)

func main() {
	switch kingpin.MustParse(app.Parse(os.Args[1:])) {
	case serverCommand.FullCommand():
		webserver := SetupWebserver()

		baseURL := "http://127.0.0.1:" + strconv.Itoa(*serverPort)
		fmt.Println(baseURL + "/hsicomponents | Get a list of HSI components")
		fmt.Println(baseURL + "/hsicomponents/quote | Get quotes for all HSI components stocks")
		fmt.Println(baseURL + "/quote/<symbol> | Supported symbol format: 700, 00700, 700:HK, etc.>")

		webserver.Start(":" + strconv.Itoa(*serverPort))
	}
}
