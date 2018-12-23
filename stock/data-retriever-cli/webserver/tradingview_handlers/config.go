package tradingview_handlers

import (
	"github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/database"
	"github.com/labstack/echo"
)

type ConfigResponse struct {
	SupportedResolutions   []string `json:"supported_resolutions"`
	SupportsGroupRequest   bool     `json:"supports_group_request"`
	SupportsMarks          bool     `json:"supports_marks"`
	SupportsSearch         bool     `json:"supports_search"`
	SupportsTimescaleMarks bool     `json:"supports_timescale_marks"`
}

func ConfigHandler(c echo.Context) error {
	/*
		"supported_resolutions": [
		   "1",
		   "5",
		   "15",
		   "30",
		   "60",
		   "1D",
		   "1W",
		   "1M"
		],
	*/

	resp := ConfigResponse{}

	result, err := database.Query("select DISTINCT(resolution) from ohlc;")
	if err != nil {
		return c.JSON(404, nil)
	}
	resolutions := []string{}
	for _, row := range result {
		if tvRepresentation, ok := DBResolutionToTVResolutionMap[row["resolution"]]; ok {
			resolutions = append(resolutions, tvRepresentation)
		}
	}

	resp.SupportedResolutions = resolutions
	resp.SupportsGroupRequest = false
	resp.SupportsMarks = false
	resp.SupportsSearch = true
	resp.SupportsTimescaleMarks = false

	return c.JSON(200, resp)
}
