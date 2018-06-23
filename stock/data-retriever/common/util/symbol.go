package util

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
)

func NewStandardSymbol(symbol string) models.StandardSymbol {
	result := models.StandardSymbol{}

	var err error
	if result.Symbol, err = ConvertToStandardSymbol(symbol); err != nil {
		panic(err)
	}

	return result
}

// ConvertToStandardSymbol converts non-standard symbol(eg: 700, 0700, 700.HK) to standard symbol(eg: 00700.HK), i.e. <5 digits stock code>.HK
func ConvertToStandardSymbol(symbol string) (string, error) {
	code, err := ExtractStockCode(symbol)
	if err != nil {
		return "", err
	}
	stockCode := StringToInt(code)

	return fmt.Sprintf("%05d.HK", stockCode), nil
}

func ExtractStockCode(str string) (string, error) {
	var re = regexp.MustCompile(`(?m)([1-9][0-9]*)`)
	match := re.FindStringSubmatch(str)
	if len(match) != 2 {
		return "", errors.New("Unable To Extract Stock Code From String: " + str)
	}
	return match[1], nil
}
