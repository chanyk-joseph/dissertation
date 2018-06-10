package util

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

func FormatRequest(r *http.Request) string {
	// Create return string
	var request []string
	// Add the request string
	url := fmt.Sprintf("%v %v %v", r.Method, r.URL, r.Proto)
	request = append(request, url)
	// Add the host
	request = append(request, fmt.Sprintf("Host: %v", r.Host))
	// Loop through headers
	for name, headers := range r.Header {
		name = strings.ToLower(name)
		for _, h := range headers {
			request = append(request, fmt.Sprintf("%v: %v", name, h))
		}
	}

	// If this is a POST, add post data
	if r.Method == "POST" {
		r.ParseForm()
		request = append(request, "\n")
		request = append(request, r.Form.Encode())
	}
	// Return the request as a string
	return strings.Join(request, "\n")
}

func StringToFloat32(str string) float32 {
	value, err := strconv.ParseFloat(str, 32)
	if err != nil {
		panic(err)
	}
	return float32(value)
}

func StringToInt(str string) int {
	value, err := strconv.Atoi(str)
	if err != nil {
		panic(err)
	}
	return value
}

func ObjectToJsonString(obj interface{}) string {
	buf, err := json.MarshalIndent(obj, "", "	")
	if err != nil {
		panic(err)
	}

	return string(buf)
}
