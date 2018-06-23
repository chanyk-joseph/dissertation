package utils

import (
	"encoding/json"
	"strconv"
)

func StringToFloat32(str string) float32 {
	value, err := strconv.ParseFloat(str, 32)
	if err != nil {
		panic(err)
	}
	return float32(value)
}

func StringToFloat64(str string) float64 {
	value, err := strconv.ParseFloat(str, 64)
	if err != nil {
		panic(err)
	}
	return value
}

func StringToInt(str string) int {
	value, err := strconv.Atoi(str)
	if err != nil {
		panic(err)
	}
	return value
}

func StringToInt64(str string) int64 {
	value, err := strconv.ParseInt(str, 10, 64)
	if err != nil {
		panic(err)
	}
	return value
}

func ObjectToJSONString(obj interface{}) string {
	buf, err := json.MarshalIndent(obj, "", "	")
	if err != nil {
		panic(err)
	}

	return string(buf)
}
