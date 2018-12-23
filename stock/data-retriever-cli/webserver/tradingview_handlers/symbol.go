package tradingview_handlers

import (
	"fmt"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/database"
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
	stockSymbol := c.FormValue("symbol")

	result, err := database.Query(fmt.Sprintf("select DISTINCT(resolution) from ohlc WHERE asset_name = '%s';", stockSymbol))
	if err != nil {
		return c.JSON(404, nil)
	}
	availableResolutions := []string{}
	for _, row := range result {
		if tvRepresentation, ok := DBResolutionToTVResolutionMap[row["resolution"]]; ok {
			availableResolutions = append(availableResolutions, tvRepresentation)
		}
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
		SupportedResolution: availableResolutions,
		PriceScale:          100,
		Ticker:              stockSymbol,
	}

	return c.JSON(200, resp)
}
