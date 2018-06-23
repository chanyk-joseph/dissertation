package investtab

import (
	"encoding/json"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
)

type Fundamentals struct {
	Symbol string `json:"symbol"`

	Close         float64 `json:"last_close"`
	MarketCapital float64 `json:"market_cap"`
	NAV           float64 `json:"last_nav"`
	Beta250Days   float64 `json:"beta_250d"`
	PriceBook     float64 `json:"last_price_book"`
	EPS           float64 `json:"last_eps"`
	PE            float64 `json:"last_pe"`
	PEG           float64 `json:"last_peg"`
	Yield         float64 `json:"last_yield"`

	IssueCapital              float64 `json:"issue_cap"`
	TotalIssueCapital         float64 `json:"total_issue_cap"`
	DividendPerShare          float64 `json:"last_dps"`
	TradingCurrency           string  `json:"trading_currency"`
	ParCurrency               string  `json:"par_currency"`
	ParValue                  float64 `json:"par_value"`
	CurrentRatio              float64 `json:"last_current_ratio"`
	AuthCapitalShares         float64 `json:"auth_cap_shares"`
	LongTermDebtToEquityRatio float64 `json:"last_ltd_equity"`
	ReturnOfEquity            float64 `json:"last_roe"`
	BoardLot                  float64 `json:"board_lot"`
	GrossMargin               float64 `json:"last_gross_margin"`
}

func (fundamentals Fundamentals) ToJSONString() string {
	return util.ObjectToJSONString(fundamentals)
}
func GetFundamentals(standardSymbol models.StandardSymbol) (Fundamentals, error) {
	var result Fundamentals
	symbol := strings.Replace(standardSymbol.Symbol, ".", ":", -1)

	urlStr := "https://api.investtab.com/api/quote/" + symbol + "/fundamentals"
	headers := map[string]string{
		"Accept":  "application/json, text/plain, */*",
		"Referer": "https://www.investtab.com/en/filter",
	}
	_, bodyStr, err := util.HttpGetResponseContentWithHeaders(urlStr, headers)
	if err != nil {
		return result, err
	}

	err = json.Unmarshal([]byte(bodyStr), &result)
	if err != nil {
		return result, err
	}

	return result, nil
}
