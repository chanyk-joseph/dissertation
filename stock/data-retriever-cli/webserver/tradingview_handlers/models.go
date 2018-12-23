package tradingview_handlers

var DBResolutionToTVResolutionMap = map[string]string{
	"minute":    "1",
	"5minutes":  "5",
	"15minutes": "15",
	"30minutes": "30",
	"hour":      "60",
	"day":       "D",
}

var TVResolutionToDBResolutionMap = map[string]string{
	"1":  "minute",
	"5":  "5minutes",
	"15": "15minutes",
	"30": "30minutes",
	"60": "hour",
	"D":  "day",
}
