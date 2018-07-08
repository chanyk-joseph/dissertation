package utils

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"time"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/bloomberg"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/hkex"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/investtab"
	"github.com/karlseguin/ccache"
	"github.com/vjeantet/jodaTime"

	"database/sql"

	_ "github.com/go-sql-driver/mysql"
)

var cache *ccache.Cache

func init() {
	cache = ccache.New(ccache.Configure())
}

type TranslateFunction func(interface{}) interface{}

type quote struct {
	Standard models.StandardQuoteFromAllProviders
	Raw      models.RawQuoteFromAllProviders
}

func ArrToUDF(inputArr interface{}, customFieldName map[string]string, customTranslateFunc map[string]TranslateFunction) (result map[string]interface{}) {
	result = map[string]interface{}{}
	defer func() { //catch or finally
		if err := recover(); err != nil {
			result["s"] = "error"
			result["errmsg"] = err
		}
	}()

	tmp := reflect.ValueOf(inputArr)
	if tmp.Kind() != reflect.Slice {
		panic("inputArr is not a array")
	}
	arr := make([]interface{}, tmp.Len())
	for i := 0; i < tmp.Len(); i++ {
		arr[i] = tmp.Index(i).Interface()
	}

	if len(arr) == 0 {
		result["s"] = "error"
		result["error"] = "Cannot convert to UDF format because no element in the array"
		return result
	}

	result["s"] = "ok"
	for k, v := range customFieldName {
		tmpArr := []interface{}{}
		translateFunc, hasTranslateFunc := customTranslateFunc[k]
		for _, obj := range arr {
			r := reflect.ValueOf(obj)
			f := reflect.Indirect(r).FieldByName(k)
			if hasTranslateFunc {
				tmpArr = append(tmpArr, translateFunc(f.Interface()))
				continue
			}
			tmpArr = append(tmpArr, f.Interface())
		}
		result[v] = tmpArr
	}
	return result
}

func HasFolder(path string) bool {
	targetFolder, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	if fi, err := os.Stat(targetFolder); err == nil && fi.IsDir() {
		return true
	}
	return false
}

func HasFile(path string) bool {
	targetFile, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	if fi, err := os.Stat(targetFile); err == nil && !fi.IsDir() {
		return true
	}
	return false
}

func CreateFolderIfNotExist(path string) error {
	if !HasFolder(path) {
		return os.Mkdir(path, os.ModePerm)
	}
	return nil
}

func GetQuoteFromAllProviders(symbol models.StandardSymbol) (models.RawQuoteFromAllProviders, models.StandardQuoteFromAllProviders, error) {
	item, err := cache.Fetch(symbol.Symbol, 2*time.Minute, func() (interface{}, error) {
		result := quote{}
		result.Standard = models.StandardQuoteFromAllProviders{}
		result.Standard.StandardSymbol = symbol
		result.Standard.Quotes = make(map[string]models.StandardQuote)
		result.Raw = models.RawQuoteFromAllProviders{}
		result.Raw.StandardSymbol = symbol
		result.Raw.Quotes = make(map[string]interface{})

		mLock := make(chan bool, 1)
		setQuote := func(providerId string, rawQuote interface{}, standardQuote models.StandardQuote) {
			mLock <- true
			result.Raw.Quotes[providerId] = rawQuote
			result.Standard.Quotes[providerId] = standardQuote
			<-mLock
		}

		var quoteWaitGroup sync.WaitGroup
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := hkex.Quote(symbol); err == nil {
				setQuote("hkex", q, hkex.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := aastocks.Quote(symbol); err == nil {
				setQuote("aastocks", q, aastocks.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := bloomberg.Quote(symbol); err == nil {
				setQuote("bloomberg", q, bloomberg.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()
		quoteWaitGroup.Add(1)
		go func() {
			if q, err := investtab.Quote(symbol); err == nil {
				setQuote("investtab", q, investtab.ToStandardQuote(q))
			}
			quoteWaitGroup.Done()
		}()

		quoteWaitGroup.Wait()
		if len(result.Standard.Quotes) == 0 {
			return nil, errors.New("No Quote From All Providers")
		}
		return result, nil
	})

	if err != nil {
		return models.RawQuoteFromAllProviders{}, models.StandardQuoteFromAllProviders{}, err
	}
	return item.Value().(quote).Raw, item.Value().(quote).Standard, nil
}

func GetQuotesOfHSIComponents() (result models.QuotesOfHSIComponents, err error) {
	result = models.QuotesOfHSIComponents{}
	hsiComponentsSymbols, err := aastocks.GetHSIConstituentsCodes()
	if err != nil {
		return result, err
	}

	for _, sym := range hsiComponentsSymbols {
		fmt.Println("Quoting " + sym.Symbol)
		_, q, err := GetQuoteFromAllProviders(sym)
		if err != nil {
			return result, err
		}
		result.Quotes = append(result.Quotes, q)
	}

	return result, nil
}

func UpdateDatabase(dbAddr string, username string, password string, dbName string) {
	defer func() {
		if recover() != nil {
			return
		}
	}()

	connectStr := fmt.Sprintf("%s:%s@tcp(%s)/%s", username, password, dbAddr, dbName)
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
