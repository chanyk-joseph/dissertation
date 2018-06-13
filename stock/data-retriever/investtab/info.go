package investtab

import (
	"encoding/json"
	"fmt"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
)

type Name struct {
	SimplifiedChinese  string `json:"sc"`
	TraditionalChinese string `json:"tc"`
	English            string `json:"en"`
}
type Sector struct {
	LastUpdated string   `json:"last_updated"`
	Name        Name     `json:"name"`
	Exchange    string   `json:"exchange"`
	Symbols     []string `json:"symbols"`
	ClassCode   string   `json:"class_code"`
	ClassKey    string   `json:"class_key"`
	ClassType   string   `json:"class_type"`
}
type Industry Sector
type SubIndustry Sector

type Info struct {
	Symbol      string                 `json:"symbol"`
	Name        Name                   `json:"names"`
	Sector      map[string]Sector      `json:"sector"`
	Industry    map[string]Industry    `json:"industry"`
	SubIndustry map[string]SubIndustry `json:"sub_industry"`
	EnabledTabs struct {
		CompanyInfo        bool `json:"company_info"`
		ShortSelling       bool `json:"shortselling"`
		FinancialStatement bool `json:"fin_stmts"`
		VWAP               bool `json:"vwap"` // volume-weighted average price
		FinancialRatios    bool `json:"fin_ratios"`
		CompanyBackground  bool `json:"company_background"`
		EarnSummary        bool `json:"earn_summary"`
		DividendHistory    bool `json:"dvd_hist"`
	} `json:"enabled_tabs"`

	InstrumentClass string  `json:"instrument_class"`
	TradingCurrency string  `json:"trading_currency"`
	Exchange        string  `json:"exchange"`
	BoardAmount     float32 `json:"board_amount"`
	BoardLot        float32 `json:"board_lot"`
	ListingDate     string  `json:"listing_date"`
	FinancialYear   struct {
		Min int `json:"min"`
		Max int `json:"max"`
	} `json:"fin_year"`
	ParCurrency    string `json:"par_currency"`
	StockType      string `json:"stock_type"`
	SecuritiesType string `json:"securities_type"`
	Shortable      bool   `json:"shortable"`
	Suspended      bool   `json:"suspended"`
}

func (info Info) ToJSONString() string {
	return util.ObjectToJSONString(info)
}

func GetInfo(symbol string) (Info, error) {
	var result Info

	urlStr := "https://api.investtab.com/api/quote/" + symbol + "/info"
	_, bodyStr, err := util.HttpGetResponseContent(urlStr)
	if err != nil {
		return result, err
	}
	if err = json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return result, err
	}

	fmt.Println(result.ToJSONString())

	return result, nil
}

/*
Example JSON:
https://api.investtab.com/api/quote/00700:HK/info
{
	"sector": {
		"07": {
			"last_updated": "2018-06-13T12:05:04.319649",
			"name": {
				"sc": "资讯科技",
				"en": "Information Technology",
				"tc": "資訊科技"
			},
			"exchange": "HKEX",
			"symbols": ["00046:HK", "00085:HK", "00092:HK", "00099:HK", "00110:HK", "00136:HK", "00139:HK", "00143:HK", "00223:HK", "00241:HK", "00243:HK", "00248:HK", "00250:HK", "00268:HK", "00285:HK", "00303:HK", "00327:HK", "00354:HK", "00395:HK", "00400:HK", "00402:HK", "00418:HK", "00434:HK", "00465:HK", "00469:HK", "00479:HK", "00484:HK", "00522:HK", "00529:HK", "00536:HK", "00543:HK", "00553:HK", "00569:HK", "00572:HK", "00595:HK", "00596:HK", "00698:HK", "00700:HK", "00724:HK", "00763:HK", "00771:HK", "00772:HK", "00777:HK", "00799:HK", "00802:HK", "00818:HK", "00854:HK", "00856:HK", "00861:HK", "00862:HK", "00877:HK", "00885:HK", "00903:HK", "00947:HK", "00948:HK", "00981:HK", "00992:HK", "01010:HK", "01013:HK", "01022:HK", "01037:HK", "01039:HK", "01050:HK", "01059:HK", "01063:HK", "01075:HK", "01079:HK", "01085:HK", "01087:HK", "01089:HK", "01094:HK", "01155:HK", "01202:HK", "01213:HK", "01236:HK", "01263:HK", "01297:HK", "01300:HK", "01337:HK", "01347:HK", "01357:HK", "01362:HK", "01385:HK", "01415:HK", "01450:HK", "01460:HK", "01478:HK", "01522:HK", "01523:HK", "01588:HK", "01613:HK", "01617:HK", "01665:HK", "01686:HK", "01708:HK", "01729:HK", "01808:HK", "01900:HK", "01933:HK", "01980:HK", "01985:HK", "02000:HK", "02018:HK", "02022:HK", "02028:HK", "02038:HK", "02086:HK", "02100:HK", "02166:HK", "02203:HK", "02239:HK", "02280:HK", "02308:HK", "02336:HK", "02342:HK", "02363:HK", "02369:HK", "02382:HK", "02708:HK", "02878:HK", "03315:HK", "03335:HK", "03336:HK", "03355:HK", "03638:HK", "03738:HK", "03773:HK", "03777:HK", "03888:HK", "03997:HK", "06036:HK", "06088:HK", "06133:HK", "06168:HK", "06188:HK", "06869:HK", "06899:HK", "08006:HK", "08013:HK", "08016:HK", "08018:HK", "08033:HK", "08043:HK", "08045:HK", "08048:HK", "08050:HK", "08051:HK", "08060:HK", "08062:HK", "08065:HK", "08071:HK", "08076:HK", "08081:HK", "08083:HK", "08086:HK", "08092:HK", "08100:HK", "08103:HK", "08106:HK", "08109:HK", "08129:HK", "08131:HK", "08148:HK", "08159:HK", "08165:HK", "08167:HK", "08171:HK", "08178:HK", "08192:HK", "08205:HK", "08206:HK", "08227:HK", "08229:HK", "08231:HK", "08236:HK", "08242:HK", "08245:HK", "08249:HK", "08255:HK", "08257:HK", "08266:HK", "08267:HK", "08282:HK", "08286:HK", "08287:HK", "08290:HK", "08311:HK", "08319:HK", "08325:HK", "08342:HK", "08345:HK", "08353:HK", "08355:HK", "08361:HK", "08379:HK", "08410:HK", "08420:HK", "08465:HK", "08487:HK", "08490:HK"],
			"class_code": "07",
			"class_key": "information-technology",
			"class_type": "sector"
		}
	},
	"trading_currency": "HKD",
	"sub_industry": {
		"070201": {
			"last_updated": "2018-06-13T12:06:02.132806",
			"name": {
				"sc": "电子商贸及互联网服务",
				"en": "e-Commerce & Internet Services",
				"tc": "電子商貿及互聯網服務"
			},
			"exchange": "HKEX",
			"symbols": ["00136:HK", "00223:HK", "00241:HK", "00395:HK", "00400:HK", "00536:HK", "00543:HK", "00572:HK", "00700:HK", "00772:HK", "01039:HK", "01075:HK", "01094:HK", "01357:HK", "01686:HK", "01980:HK", "02280:HK", "08006:HK", "08033:HK", "08076:HK", "08083:HK", "08086:HK", "08255:HK", "08325:HK", "08361:HK"],
			"class_code": "070201",
			"class_key": "e-commerce-internet-services",
			"class_type": "subindustry"
		}
	},
	"enabled_tabs": {
		"company_info": true,
		"shortselling": true,
		"fin_stmts": true,
		"vwap": true,
		"fin_ratios": true,
		"company_background": true,
		"earn_summary": true,
		"dvd_hist": true
	},
	"board_amount": 41520.0,
	"symbol": "00700:HK",
	"par_currency": "HKD",
	"stock_type": "O",
	"names": {
		"sc": "腾讯控股",
		"en": "Tencent Hold",
		"tc": "騰訊控股"
	},
	"suspended": false,
	"shortable": true,
	"securities_type": "OS",
	"fin_year": {
		"max": 2017,
		"min": 2004
	},
	"parallel_trading": null,
	"listing_date": "2004-06-16T00:00:00",
	"industry": {
		"0702": {
			"last_updated": "2018-06-13T12:06:00.460727",
			"name": {
				"sc": "软件服务",
				"en": "Software & Services",
				"tc": "軟件服務"
			},
			"exchange": "HKEX",
			"symbols": ["00046:HK", "00092:HK", "00136:HK", "00223:HK", "00241:HK", "00250:HK", "00268:HK", "00354:HK", "00395:HK", "00400:HK", "00402:HK", "00418:HK", "00434:HK", "00484:HK", "00536:HK", "00543:HK", "00569:HK", "00572:HK", "00596:HK", "00700:HK", "00771:HK", "00772:HK", "00777:HK", "00799:HK", "00802:HK", "00818:HK", "00861:HK", "00862:HK", "01013:HK", "01022:HK", "01037:HK", "01039:HK", "01059:HK", "01075:HK", "01089:HK", "01094:HK", "01236:HK", "01297:HK", "01357:HK", "01450:HK", "01460:HK", "01522:HK", "01588:HK", "01613:HK", "01665:HK", "01686:HK", "01708:HK", "01808:HK", "01900:HK", "01933:HK", "01980:HK", "01985:HK", "02022:HK", "02100:HK", "02280:HK", "02708:HK", "03738:HK", "03888:HK", "06899:HK", "08006:HK", "08013:HK", "08018:HK", "08033:HK", "08045:HK", "08048:HK", "08060:HK", "08062:HK", "08065:HK", "08071:HK", "08076:HK", "08081:HK", "08083:HK", "08086:HK", "08092:HK", "08100:HK", "08103:HK", "08106:HK", "08109:HK", "08129:HK", "08131:HK", "08148:HK", "08165:HK", "08178:HK", "08205:HK", "08206:HK", "08229:HK", "08236:HK", "08249:HK", "08255:HK", "08267:HK", "08282:HK", "08290:HK", "08319:HK", "08325:HK", "08342:HK", "08345:HK", "08353:HK", "08355:HK", "08361:HK", "08379:HK", "08420:HK"],
			"class_code": "0702",
			"class_key": "software-services",
			"class_type": "industry"
		}
	},
	"exchange": "HKEX",
	"board_lot": 100.0,
	"instrument_class": "stock"
}
*/
