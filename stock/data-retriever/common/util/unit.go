package util

import (
	"strings"
)

func ConvertNumberWithUnitToActualNumber(unit string) float64 {
	unit = strings.ToUpper(unit)
	unit = strings.Replace(unit, ",", "", -1)
	unit = strings.Replace(unit, " ", "", -1)

	if i := strings.Index(unit, "K"); i >= 0 {
		unit = strings.Replace(unit, "K", "", -1)
		return StringToFloat64(unit) * 1000
	} else if i := strings.Index(unit, "M"); i >= 0 {
		unit = strings.Replace(unit, "M", "", -1)
		return StringToFloat64(unit) * 1000000
	} else if i := strings.Index(unit, "B"); i >= 0 {
		unit = strings.Replace(unit, "B", "", -1)
		return StringToFloat64(unit) * 1000000000
	} else if i := strings.Index(unit, "T"); i >= 0 {
		unit = strings.Replace(unit, "T", "", -1)
		return StringToFloat64(unit) * 1000000000000
	}

	return StringToFloat64(unit)
}
