package hkex

import (
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/pkg/errors"
)

func getAccessToken() (string, error) {
	urlStr := "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=en"

	_, bodyString, err := utils.HttpGetResponseContent(urlStr)
	if err != nil {
		return "", err
	}

	re := regexp.MustCompile(`(?m)Base64-AES-Encrypted-Token"[\s\S]*?"(.*?)";`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return "", errors.New("Unable To Locate Access Key From Response: \n" + bodyString)
	}

	return match[1], nil
}
