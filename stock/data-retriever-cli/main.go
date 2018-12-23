//go:generate fileb0x.exe b0x.json
package main

import (
	"bufio"
	"database/sql"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/database"
	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver"
	"github.com/vjeantet/jodaTime"
	"gopkg.in/alecthomas/kingpin.v2"
)

var (
	app               = kingpin.New("data-retriever", "Serve stock quote via RestAPI / save to mysql database")
	serverCommand     = app.Command("web", "Web Server Mode")
	serverPort        = serverCommand.Flag("port", "Port For Listening Slave Machine, Default = 8888").Default("8888").Short('p').Int()
	tradingviewDBAddr = serverCommand.Flag("db-address", "DB address").Default("dissertation.czjxnnexr3c6.ap-northeast-1.rds.amazonaws.com:3306").String()
	tradingviewDBUser = serverCommand.Flag("db-user", "DB username").Default("chanykjoseph").String()
	tradingviewDBPW   = serverCommand.Flag("db-pw", "DB password").Default("Test1234").String()

	dbUpdateCommand         = app.Command("update", "Update mysql database with all HSI components quotes. If the table 'stocks_quotes' does not exist, it will create one")
	dbServerAddr            = dbUpdateCommand.Arg("ip", "Mysql Server IP").Required().String()
	dbName                  = dbUpdateCommand.Arg("db", "Database name").Required().String()
	dbUsername              = dbUpdateCommand.Arg("username", "Username for Mysql Server").Required().String()
	dbPassword              = dbUpdateCommand.Arg("password", "Password for Mysql Server").Required().String()
	updateIntervalInSeconds = dbUpdateCommand.Flag("interval", "Update interval in seconds, Default = 60").Default("60").Short('i').Int()

	importOHLC  = app.Command("import-ohlc", "Import OHLC csv, time,open,high,low,close,volume")
	csvFilePath = importOHLC.Arg("csv", "Input CSV File Path").Required().String()
	assetName   = importOHLC.Arg("asset-name", "Asset Name").Required().String()
	resolution  = importOHLC.Arg("resolution", "Resolution").Required().String()

	importTick      = app.Command("import-tick", "Import tick csv, time,bid,ask")
	tickCSVFilePath = importTick.Arg("csv", "Input CSV File Path").Required().String()
	tickAssetName   = importTick.Arg("asset-name", "Asset Name").Required().String()
)

func main() {
	switch kingpin.MustParse(app.Parse(os.Args[1:])) {
	case serverCommand.FullCommand():
		webServer := webserver.SetupWebserver()

		baseURL := "http://127.0.0.1:" + strconv.Itoa(*serverPort)
		fmt.Println(baseURL + "/hsicomponents | Get a list of HSI components")
		fmt.Println(baseURL + "/hsicomponents/quote | Get quotes for all HSI components stocks")
		fmt.Println(baseURL + "/quote/<symbol> | Supported symbol format: 700, 00700, 700:HK, etc.>")
		fmt.Println(baseURL + "/chart/index.html | Stock Chart Using TradingView")

		database.SetupDatabase(*tradingviewDBAddr, *tradingviewDBUser, *tradingviewDBPW)
		webServer.Start(":" + strconv.Itoa(*serverPort))
	case dbUpdateCommand.FullCommand():
		fmt.Println("Update Interval: " + strconv.Itoa(*updateIntervalInSeconds) + "s")
		ticker := time.NewTicker(time.Duration(*updateIntervalInSeconds) * time.Second)
		go func() {
			for {
				select {
				case <-ticker.C:
					fmt.Println("Start Updating Database")
					CLIUtils.UpdateDatabase(*dbServerAddr, *dbUsername, *dbPassword, *dbName)
				}
			}
		}()
		fmt.Println("Press the Enter Key To Stop")
		var input string
		fmt.Scanln(&input)
	case importOHLC.FullCommand():
		connectStr := fmt.Sprintf("%s:%s@tcp(%s)/%s", "chanykjoseph", "Test1234", "dissertation.czjxnnexr3c6.ap-northeast-1.rds.amazonaws.com:3306", "dissertation")
		db, err := sql.Open("mysql", connectStr)
		defer db.Close()
		if err != nil {
			log.Println(err.Error())
			return
		}

		createTableSQLStr := "CREATE TABLE  IF NOT EXISTS `dissertation`.`ohlc` (" +
			"`asset_name` varchar(20) NOT NULL," +
			"`resolution` varchar(20) NOT NULL," +
			"`datetime` datetime NOT NULL," +
			"`open` double unsigned NOT NULL," +
			"`high` double unsigned NOT NULL," +
			"`low` double unsigned NOT NULL," +
			"`close` double unsigned NOT NULL," +
			"`volume` double unsigned NOT NULL," +
			"PRIMARY KEY (`asset_name`, `resolution`, `datetime`));"
		db.Exec(createTableSQLStr)

		fmt.Println("Read CSV: " + *csvFilePath)
		f, e := os.Open(*csvFilePath)
		if e != nil {
			panic(e)
		}
		tx, _ := db.Begin()
		var headers []string
		r := csv.NewReader(bufio.NewReader(f))
		for j := 0; true; j++ {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if j == 0 && len(record) > 0 {
				headers = record
			} else {
				//Ignore row if the number of column != num of headers
				if len(record) == len(headers) {
					t, err := jodaTime.Parse("YYYYMMdd HH:mm:ss", record[0][:len(record[0])-4])
					if err != nil {
						panic(err)
					}
					open, _ := strconv.ParseFloat(record[1], 64)
					high, _ := strconv.ParseFloat(record[2], 64)
					low, _ := strconv.ParseFloat(record[3], 64)
					close, _ := strconv.ParseFloat(record[4], 64)
					volume, _ := strconv.ParseFloat(record[5], 64)
					sqlStr := fmt.Sprintf("INSERT INTO `dissertation`.`ohlc` (`asset_name`, `resolution`, `datetime`, `open`, `high`, `low`, `close`, `volume`) VALUES ('%s', '%s', from_unixtime(%d), %.6f, %.6f, %.6f, %.6f, %.6f) ON DUPLICATE KEY UPDATE `open`=VALUES(`open`), `high`=VALUES(`high`), `low`=VALUES(`low`), `close`=VALUES(`close`), `volume`=VALUES(`volume`)", *assetName, *resolution, t.UTC().Unix(), open, high, low, close, volume)
					tx.Exec(sqlStr)
					fmt.Println(j)
				}
			}
		}
		fmt.Println("Commit")
		if err := tx.Commit(); err != nil {
			panic(err)
		}
	case importTick.FullCommand():
		connectStr := fmt.Sprintf("%s:%s@tcp(%s)/%s", "chanykjoseph", "Test1234", "dissertation.czjxnnexr3c6.ap-northeast-1.rds.amazonaws.com:3306", "dissertation")
		db, err := sql.Open("mysql", connectStr)
		defer db.Close()
		if err != nil {
			log.Println(err.Error())
			return
		}

		createTableSQLStr := "CREATE TABLE IF NOT EXISTS `dissertation`.`tick` (" +
			"`asset_name` varchar(20) NOT NULL," +
			"`date` date NOT NULL," +
			"`datetime` datetime(6) NOT NULL," +
			"`bid_price` double unsigned NOT NULL," +
			"`ask_price` double unsigned NOT NULL," +
			"PRIMARY KEY (`asset_name`, `date`, `datetime`));"
		_, err = db.Exec(createTableSQLStr)
		if err != nil {
			panic(err)
		}

		fmt.Println("Read CSV: " + *tickCSVFilePath)
		f, e := os.Open(*tickCSVFilePath)
		if e != nil {
			panic(e)
		}
		tx, _ := db.Begin()
		var headers []string
		r := csv.NewReader(bufio.NewReader(f))
		for j := 0; true; j++ {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if j == 0 && len(record) > 0 {
				headers = record
			} else {
				//Ignore row if the number of column != num of headers
				if len(record) == len(headers) {
					d := record[0]
					year := d[0:4]
					month := d[4:6]
					day := d[6:8]
					date := year + "-" + month + "-" + day
					datetime := date + " " + d[9:17] + "." + d[18:]
					bid, _ := strconv.ParseFloat(record[1], 64)
					ask, _ := strconv.ParseFloat(record[2], 64)
					sqlStr := fmt.Sprintf("INSERT INTO `dissertation`.`tick` (`asset_name`, `date`, `datetime`, `bid_price`, `ask_price`) VALUES ('%s', '%s', '%s', %.6f, %.6f) ON DUPLICATE KEY UPDATE `bid_price`=VALUES(`bid_price`), `ask_price`=VALUES(`ask_price`)", *tickAssetName, date, datetime, bid, ask)
					_, e := tx.Exec(sqlStr)
					if e != nil {
						panic(e)
					}
				}
			}
			if j%100000 == 0 {
				fmt.Println("Commit")
				fmt.Println(j)
				if err := tx.Commit(); err != nil {
					panic(err)
				}
				tx, _ = db.Begin()
			}
		}
		fmt.Println("Commit")
		if err := tx.Commit(); err != nil {
			panic(err)
		}
	}
}
