package tradingview_handlers

import (
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/webserver/common"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
)

type SymbolResponse struct {
	Name                string   `json:"name"`
	ExchangeTraded      string   `json:"exchange-traded"`
	ExchangeListed      string   `json:"exchange-listed"`
	Timezone            string   `json:"timezone"`
	MinMov              int      `json:"minmov"`
	MinMov2             int      `json:"minmov2"`
	PointValue          int      `json:"pointvalue"`
	Session             string   `json:"session"`
	HasIntraday         bool     `json:"has_intraday"`
	HasNoVolume         bool     `json:"has_no_volume"`
	Description         string   `json:"description"`
	Type                string   `json:"type"`
	SupportedResolution []string `json:"supported_resolution"`
	PriceScale          int      `json:"pricescale"`
	Ticker              string   `json:"ticker"`
}

func SymbolHandler(c echo.Context) error {
	stockSymbol, err := utils.ConvertToStandardSymbol(c.FormValue("symbol"))
	if err != nil {
		return c.JSON(500, common.ErrorWrapper{err.Error()})
	}

	resp := SymbolResponse{
		Name:                stockSymbol,
		ExchangeTraded:      "HSI",
		ExchangeListed:      "HSI",
		Timezone:            "Asia/Hong_Kong",
		MinMov:              1,
		MinMov2:             0,
		PointValue:          1,
		Session:             "0930-1600",
		HasIntraday:         true,
		HasNoVolume:         false,
		Description:         stockSymbol,
		Type:                "stock",
		SupportedResolution: []string{"D", "2D", "3D", "W", "3W", "M", "6M"},
		PriceScale:          100,
		Ticker:              stockSymbol,
	}

	return c.JSON(200, resp)
}
