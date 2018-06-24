package main

import (
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
	"github.com/vjeantet/jodaTime"

	"github.com/karlseguin/ccache"

	"database/sql"

	_ "github.com/go-sql-driver/mysql"
)

var cache *ccache.Cache

func init() {
	cache = ccache.New(ccache.Configure())
}

func GetQuoteFromAllProviders(symbol models.StandardSymbol) (models.QuoteFromAllProviders, error) {
	item, err := cache.Fetch(symbol.Symbol, 2*time.Minute, func() (interface{}, error) {
		result := models.QuoteFromAllProviders{}
		result.StandardSymbol = symbol
		result.Quotes = make(map[string]models.StandardQuote)

		mLock := make(chan bool, 1)
		setQuote := func(providerId string, quote models.StandardQuote) {
			mLock <- true
			result.Quotes[providerId] = quote
			<-mLock
		}

		var quoteWaitGroup sync.WaitGroup
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := hkex.Quote(symbol); err == nil {
				setQuote("hkex", hkex.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := aastocks.Quote(symbol); err == nil {
				setQuote("aastocks", aastocks.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := bloomberg.Quote(symbol); err == nil {
				setQuote("bloomberg", bloomberg.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := investtab.Quote(symbol); err == nil {
				setQuote("investtab", investtab.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()

		quoteWaitGroup.Wait()
		if len(result.Quotes) == 0 {
			return nil, errors.New("No Quote From All Providers")
		}
		return result, nil
	})

	if err != nil {
		return models.QuoteFromAllProviders{}, err
	}
	return item.Value().(models.QuoteFromAllProviders), nil
}

func GetQuotesOfHSIComponents() (result models.QuotesOfHSIComponents, err error) {
	result = models.QuotesOfHSIComponents{}
	hsiComponentsSymbols, err := aastocks.GetHSIConstituentsCodes()
	if err != nil {
		return result, err
	}

	for _, sym := range hsiComponentsSymbols {
		fmt.Println("Quoting " + sym.Symbol)
		q, err := GetQuoteFromAllProviders(sym)
		if err != nil {
			return result, err
		}
		result.Quotes = append(result.Quotes, q)
	}

	return result, nil
}

func UpdateDatabase(dbAddr string, username string, password string) {
	connectStr := fmt.Sprintf("%s:%s@tcp(%s)/haudosicom_dissertation", username, password, dbAddr)
	db, err := sql.Open("mysql", connectStr)
	defer db.Close()
	if err != nil {
		log.Println(err.Error())
		return
	}

	createTableSQLStr := "CREATE TABLE  IF NOT EXISTS `stocks_quotes` (" +
		"`date` date NOT NULL," +
		"`symbol` varchar(20) COLLATE latin1_general_ci NOT NULL," +
		"`provider` varchar(20) COLLATE latin1_general_ci NOT NULL," +
		"`open` float NOT NULL," +
		"`low` float NOT NULL," +
		"`high` float NOT NULL," +
		"`close` float NOT NULL," +
		"`volume` int(11) NOT NULL," +
		"PRIMARY KEY (`date`, `symbol`, `provider`)" +
		") ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_general_ci;"
	db.Exec(createTableSQLStr)

	quoteOfAllHSIComponents, err := GetQuotesOfHSIComponents()
	if err != nil {
		log.Println(err.Error())
		return
	}
	tx, _ := db.Begin()
	dateStr := jodaTime.Format("YYYY-MM-dd", time.Now().UTC())
	for _, quoteFromAllProviders := range quoteOfAllHSIComponents.Quotes {
		symbol := quoteFromAllProviders.Symbol
		for providerID, quote := range quoteFromAllProviders.Quotes {
			sqlStr := fmt.Sprintf("INSERT INTO `stocks_quotes` (`date`, `symbol`, `provider`, `open`, `low`, `high`, `close`, `volume`) VALUES ('%s', '%s', '%s', %.6f, %.6f, %.6f, %.6f, %d) ON DUPLICATE KEY UPDATE `open`=VALUES(`open`), `low`=VALUES(`low`), `high`=VALUES(`high`), `close`=VALUES(`close`), `volume`=VALUES(`volume`)", dateStr, symbol, providerID, quote.Open, quote.Low, quote.High, quote.Close, quote.Volume)
			tx.Exec(sqlStr)
		}
	}
	tx.Commit()
}
