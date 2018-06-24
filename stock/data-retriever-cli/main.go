package main

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	app           = kingpin.New("data-retriever", "Serve stock quote via RestAPI / save to mysql database")
	serverCommand = app.Command("web", "Web Server Mode")
	serverPort    = serverCommand.Flag("port", "Port For Listening Slave Machine, Default = 8888").Default("8888").Short('p').Int()

	dbUpdateCommand         = app.Command("update", "Update mysql database with all HSI components quotes. If the table 'stocks_quotes' does not exist, it will create one")
	dbServerAddr            = dbUpdateCommand.Arg("ip", "Mysql Server IP").Required().String()
	dbUsername              = dbUpdateCommand.Arg("username", "Username for Mysql Server").Required().String()
	dbPassword              = dbUpdateCommand.Arg("password", "Password for Mysql Server").Required().String()
	updateIntervalInSeconds = dbUpdateCommand.Flag("interval", "Update interval in seconds, Default = 60").Default("60").Short('i').Int()
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
	case dbUpdateCommand.FullCommand():
		fmt.Println("Update Interval: " + strconv.Itoa(*updateIntervalInSeconds) + "s")
		ticker := time.NewTicker(time.Duration(*updateIntervalInSeconds) * time.Second)
		go func() {
			for {
				select {
				case <-ticker.C:
					fmt.Println("Start Updating Database")
					UpdateDatabase(*dbServerAddr, *dbUsername, *dbPassword)
				}
			}
		}()
		fmt.Println("Press the Enter Key To Stop")
		var input string
		fmt.Scanln(&input)
	}
}
