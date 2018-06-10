package hkex

import (
	"io/ioutil"
	"net/http"
	"regexp"

	"github.com/pkg/errors"
)

func getAccessToken() (string, error) {
	urlStr := "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=en"

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil)
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")

	resp, err := client.Do(r)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", errors.Errorf("Failed To Get HKEX Access Token | %s | Response Status Code: %v", urlStr, resp.StatusCode)
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	re := regexp.MustCompile(`(?m)Base64-AES-Encrypted-Token"[\s\S]*?"(.*?)";`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return "", errors.New("Unable To Locate Access Key From Response: \n" + bodyString)
	}

	return match[1], nil
}
