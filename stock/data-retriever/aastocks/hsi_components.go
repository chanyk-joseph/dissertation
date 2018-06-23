package aastocks

import (
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/models"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/pkg/errors"
)

// GetHSIConstituentsCodes returns a list of HSI Constituents Codes
// Example Equity Codes:
// [00941.HK 01038.HK 01044.HK 01088.HK 01093.HK 01109.HK 01113.HK 01299.HK 01398.HK 01928.HK 01997.HK 02007.HK 02018.HK 02318.HK 02319.HK 02382.HK 02388.HK 02628.HK 03328.HK 03988.HK 00001.HK 00002.HK 00003.HK 00005.HK 00006.HK 00011.HK 00012.HK 00016.HK 00017.HK 00019.HK 00023.HK 00027.HK 00066.HK 00083.HK 00101.HK 00144.HK 00151.HK 00175.HK 00267.HK 00288.HK 00386.HK 00388.HK 00688.HK 00700.HK 00762.HK 00823.HK 00836.HK 00857.HK 00883.HK 00939.HK]
func GetHSIConstituentsCodes() ([]models.StandardSymbol, error) {
	urlStr := "http://www.aastocks.com/en/stocks/market/index/hk-index-con.aspx"

	_, bodyString, err := util.HttpGetResponseContent(urlStr)
	if err != nil {
		return nil, err
	}

	var re = regexp.MustCompile(`(?m)detail-quote.aspx\?symbol=.*?>([0-9]{5}.HK)<\/a>`)
	matches := re.FindAllStringSubmatch(bodyString, -1)

	result := []models.StandardSymbol{}
	for _, match := range matches {
		result = append(result, models.StandardSymbol{Symbol: match[1]})
	}

	if len(result) != 50 {
		return nil, errors.Errorf("The Number of HSI Constituents Retrieved From aastocks Is Not 50 | Response:\n%s", bodyString)
	}

	return result, nil
}
