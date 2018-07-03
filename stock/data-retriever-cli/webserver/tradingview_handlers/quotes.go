package tradingview_handlers

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/aastocks"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
)

func QuotesHandler(c echo.Context) error {
	symbolsList := c.FormValue("symbols")
	symbols := strings.Split(symbolsList, ",")

	type entry struct {
		Status string `json:"s"`
		Symbol string `json:"n"`

		Quote struct {
			PriceChange           string  `json:"ch"`
			PriceChangePercentage string  `json:"chp"`
			ShortName             string  `json:"short_name"`
			Exchange              string  `json:"exchange"`
			Description           string  `json:"description"`
			LastTradedPrice       string  `json:"lp"`
			AskPrice              string  `json:"ask"`
			BidPrice              string  `json:"bid"`
			Spread                *string `json:"spread"`
			OpenPrice             string  `json:"open_price"`
			HighPrice             string  `json:"high_price"`
			LowPrice              string  `json:"low_price"`
			PreviousClosePrice    string  `json:"prev_close_price"`
			Volume                string  `json:"volume"`
		} `json:"v"`
	}
	type resp struct {
		Status   string  `json:"s"`
		ErrorMsg *string `json:"errmsg"`
		Data     []entry `json:"d"`
	}

	float2Str := func(val float64) string {
		return fmt.Sprintf("%.2f", val)
	}

	result := resp{}
	result.Status = "ok"
	for _, sym := range symbols {
		e := entry{}
		q, err := aastocks.Quote(models.StandardSymbol{sym})
		if err != nil {
			e.Status = "error"
			result.Status = "error"
			msg := *result.ErrorMsg
			msg += err.Error() + "\n"
			result.ErrorMsg = &msg
			continue
		}

		e.Status = "ok"
		e.Symbol = sym
		e.Quote.AskPrice = float2Str(q.Ask)
		e.Quote.BidPrice = float2Str(q.Bid)
		e.Quote.Description = sym
		e.Quote.Exchange = "HKEX"
		e.Quote.HighPrice = float2Str(q.High)
		e.Quote.LastTradedPrice = float2Str(q.LastTradedPrice)
		e.Quote.LowPrice = float2Str(q.Low)
		e.Quote.OpenPrice = float2Str(q.Open)
		e.Quote.PreviousClosePrice = float2Str(q.PreviousClose)
		e.Quote.PriceChange = float2Str(q.LastTradedPrice - q.PreviousClose)
		e.Quote.PriceChangePercentage = float2Str((q.LastTradedPrice - q.PreviousClose) / q.PreviousClose * 100)
		e.Quote.ShortName = sym
		e.Quote.Volume = strconv.Itoa(int(utils.ConvertNumberWithUnitToActualNumber(q.Volume)))

		result.Data = append(result.Data, e)
	}

	return c.JSON(200, result)
}
