package tradingview_handlers

import "github.com/labstack/echo"

func ConfigHandler(c echo.Context) error {
	resp := []byte(`{
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
		"supports_group_request": false,
		"supports_marks": false,
		"supports_search": true,
		"supports_timescale_marks": false
	 }`)
	return c.JSONBlob(200, resp)
}
