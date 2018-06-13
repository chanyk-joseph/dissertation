package hkex

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"regexp"

	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/util"
	"github.com/pkg/errors"
)

/*
Example JSON:
{
	"updatetime": "2018年6月11日16:09",
	"nm_s": "騰訊控股",
	"nm": "騰訊控股有限公司",
	"sym": "700",
	"ric": "0700.HK",
	"eps": 7.5986,
	"eps_ccy": "RMB",
	"pe": "44.53",
	"div_yield": "0.21",
	"mkt_cap": "3,982.21",
	"mkt_cap_u": "B",
	"ls": "419.000",
	"hc": "415.000",
	"op": "420.000",
	"hi": "421.000",
	"lo": "415.600",
	"hi52": "476.600",
	"lo52": "260.379",
	"am": "6.48",
	"am_u": "B",
	"vo": "15.48",
	"vo_u": "M",
	"bd": "418.800",
	"as": "419.000"
}
*/
type EquityQuote struct {
	UpdateTime string `json:"updatetime"`

	ShortCompanyName      string `json:"nm_s"`
	CompanyName           string `json:"nm"`
	Symbol                string `json:"sym"`
	ReutersInstrumentCode string `json:"ric"`

	EPS               float32 `json:"eps"`
	EPSCurrency       string  `json:"eps_ccy"`
	PE                string  `json:"pe"`
	DividendYield     string  `json:"div_yield"`
	MarketCapital     string  `json:"mkt_cap"`
	MarketCapitalUnit string  `json:"mkt_cap_u"`

	LastTradedPrice string `json:"ls"`
	PreviousClose   string `json:"hc"`
	Open            string `json:"op"`
	High            string `json:"hi"`
	Low             string `json:"lo"`
	High52Weeks     string `json:"hi52"`
	Low52Weeks      string `json:"lo52"`

	TurnOver     string `json:"am"`
	TurnOverUnit string `json:"am_u"`
	Volume       string `json:"vo"`
	VolumeUnit   string `json:"vo_u"`
	Bid          string `json:"bd"`
	Ask          string `json:"as"`
}

func (quote EquityQuote) ToJSONString() string {
	return util.ObjectToJsonString(quote)
}

func Quote(symbol string) (EquityQuote, error) {
	result := EquityQuote{}

	accessToken, err := getAccessToken()
	if err != nil {
		return result, err
	}
	urlStr := "https://www1.hkex.com.hk/hkexwidget/data/getequityquote?sym=" + symbol + "&token=" + accessToken + "&lang=chi&qid=1528572605481&callback=jQuery311037427382333777826_1528572604782&_=1528572604783"

	// fmt.Println(urlStr)

	client := &http.Client{}
	r, _ := http.NewRequest("GET", urlStr, nil) // URL-encoded payload
	r.Header.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
	r.Header.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.79 Safari/537.36")
	r.Header.Add("Referer", "https://www.hkex.com.hk/Market-Data/Securities-Prices/Equities?sc_lang=zh-hk")

	resp, err := client.Do(r)
	if err != nil {
		return result, err
	}
	if resp.StatusCode != http.StatusOK {
		return result, errors.Errorf("Failed To Get HKEX Access Token | %s | Response Status Code: %v", urlStr, resp.StatusCode)
	}

	bodyBytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return result, errors.Wrap(err, "Failed To Get Response Body")
	}
	bodyString := string(bodyBytes)

	re := regexp.MustCompile(`(?m)\(([\s\S]*?)\)$`)
	match := re.FindStringSubmatch(bodyString)
	if len(match) != 2 {
		return result, errors.New("Unable To Locate Stock List Array From Response: \n" + bodyString)
	}
	jsonStr := match[1]

	quoteResp := &struct {
		Data struct {
			Quote EquityQuote
		}
	}{}

	err = json.Unmarshal([]byte(jsonStr), &quoteResp)
	if err != nil {
		return result, err
	}

	if quoteResp.Data.Quote.CompanyName == "" {
		return result, errors.Errorf("Unexpected Response From HKEX\nRequest:\n%s\nResponse:\n%s", util.FormatRequest(r), jsonStr)
	}

	result = quoteResp.Data.Quote
	return result, nil
}

////////////////// Sample Quote JSON Response ////////////////////
/*
{
	"data": {
		"responsecode": "000",
		"responsemsg": "",
		"quote": {
			"hi": "426.600",
			"rs_stock_flag": false,
			"fiscal_year_end": "2017年12月31日",
			"hist_closedate": "2018年6月7日",
			"replication_method": null,
			"amt_os": "9,504,093,926",
			"primaryexch": "香港交易所",
			"ric": "0700.HK",
			"product_subtype": null,
			"db_updatetime": "2018年6月9日07:41",
			"mkt_cap_u": "B",
			"am_u": "B",
			"ew_sub_right": "",
			"secondary_listing": false,
			"ew_amt_os_cur": null,
			"ccy": "HKD",
			"management_fee": "",
			"ew_underlying_code": null,
			"trdstatus": "N",
			"nav": "",
			"original_offer_price": "",
			"issue": "",
			"asset_class": null,
			"eps": 7.5986,
			"inline_upper_strike_price": "",
			"sedol": "BMMV2K8",
			"am": "12.08",
			"iv": "",
			"ew_strike": "",
			"as": "415.000",
			"geographic_focus": null,
			"incorpin": "開曼群島",
			"ew_amt_os": "",
			"bd": "414.800",
			"registrar": "香港中央證券登記有限公司",
			"depositary": null,
			"exotic_type": null,
			"callput_indicator": null,
			"primary_market": null,
			"underlying_index": null,
			"lot": "100",
			"lo52": "260.379",
			"shares_issued_date": "2018年5月31日",
			"premium": "",
			"strike_price_ccy": null,
			"yield": "",
			"vo_u": "M",
			"base_currency": null,
			"coupon": "",
			"expiry_date": "",
			"chairman": "馬化騰",
			"underlying_ric": "0700.HK",
			"hi52": "476.600",
			"issuer_name": "騰訊控股有限公司",
			"h_share_flag": false,
			"ew_sub_per_from": "",
			"div_yield": "0.21",
			"interest_payment_date": "-",
			"updatetime": "2018年6月8日16:08",
			"aum_date": "",
			"lo": "412.600",
			"mkt_cap": "3,944.19",
			"f_aum_hkd": null,
			"ew_sub_per_to": "",
			"ls": "415.000",
			"nav_date": "",
			"csic_classification": null,
			"floating_flag": false,
			"issued_shares_note": null,
			"eff_gear": "",
			"board_lot_nominal": "",
			"hsic_ind_classification": "資訊科技業 - 軟件服務",
			"ew_desc": null,
			"inception_date": "",
			"nc": "-14.200",
			"aum": "",
			"vo": "28.93",
			"secondary_listing_flag": false,
			"listing_date": "2004年6月16日",
			"as_at_label": "截至",
			"ew_amt_os_dat": "",
			"nm": "騰訊控股有限公司",
			"nm_s": "騰訊控股",
			"sym": "700",
			"inline_lower_strike_price": "",
			"listing_category": "主要上市",
			"ew_strike_cur": null,
			"exotic_warrant_indicator": null,
			"investment_focus": null,
			"call_price": "",
			"tck": "0.200",
			"strike_price": "",
			"summary": "騰訊控股有限公司是一家主要提供增值服務及網絡廣告服務的投資控股公司。該公司通過三大分部運營。增值服務分部主要包括互聯網及移動平臺提供的網絡╱手機遊戲、社區增值服務及應用。網絡廣告分部主要包括效果廣告及展示廣告。其他分部主要包括支付相關服務、雲服務及其他服務。",
			"op": "426.600",
			"aum_u": "",
			"nav_ccy": null,
			"os": "",
			"wnt_gear": "",
			"transfer_of_listing_date": "",
			"hsic_sub_sector_classification": "電子商貿及互聯網服務",
			"domicile_country": null,
			"entitlement_ratio": "",
			"product_type": "EQTY",
			"office_address": "香港<br/>灣仔<br/>皇后大道東1號<br/>太古廣場三座29樓",
			"pc": "-3.31",
			"days_to_expiry": null,
			"underlying_code": null,
			"pe": "44.10",
			"eps_ccy": "RMB",
			"hdr": false,
			"launch_date": "",
			"hc": "429.200",
			"isin": "KYG875721634",
			"moneyness": ""
		}
	},
	"qid": "1528572605481"
}
*/

/*
//////////////////////////////////////////////////////////////////////////////////
// LabCI HKEX Widget - Common Resources
//////////////////////////////////////////////////////////////////////////////////

// Common Resource - zh_HK
LabCI.WP.CommonRC.zh_HK = {

    exchsect: {
        s: {

        },
        l: {

        }
    },
    currency: {
        'HKD': '$',
        'CNY': 'RMB',
        'USD': 'USD',
        'CNH': 'RMB'
    },
    currencydisplay: {
        'HKD': 'HK$',
        'USD': 'US$',
        'CNY': 'RMB',
        'CNH': 'RMB'
    },
    strikecurrencydisplay: {
        'HKD': 'HK$',
        'USD': 'US$',
        'CNY': 'RMB'
    },
    currencyfulldisplay: {
        'HKD': 'HKD',
        'USD': 'USD',
        'CNY': 'RMB',
        'CNH': 'RMB'
    },
    callputdisplay: {
        'Call': '認購',
        'Put': '認沽',
        'Bull': '牛證',
        'Bear': '熊證'
    },
    df: {
        FORMATPATTERN_DATETIME_MD: "M月d日 HH:mm",
        FORMATPATTERN_DATETIME_SHORTMD: "dd/MM HH:mm",
        FORMATPATTERN_DATE_MD: "M月d日",
        FORMATPATTERN_DATE_SHORTMD: "dd/MM",
        FORMATPATTERN_DATE_YMD: "d/M/yyyy",
        FORMATPATTERN_DATE_SHORTYMD: "d/M/yy",
        FORMATPATTERN_DATETIME_LONGYMD: "d/M/yyyy HH:mm:ss",
        FORMATPATTERN_DATETIME_YMD: "d/M/yyyy HH:mm",
        FORMATPATTERN_DATETIME_SHORTYMD: "d/M/yy HH:mm"
    },

    unitscale: {
        "x1000": "千",
        "x10000": "萬",
        "x10K": "萬",
        "x1000000": "百萬",
        "x1M": "百萬",
        "x100000000": "億"
    },

    msg: {
    	dn: {
    		y: '年',
    		m: '月',
    		d: '日'
    	},
        chart: {
            pclose: '上日收市',
            noData: '暫時沒有數據提供',
            months: ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'],
            short_months: ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'],
            dis_2year: '僅供下載輸出最近兩年的數據。'
        },
        interval: {
            min: '1分鐘圖',
            min5: '5分鐘圖',
            min15: '15分鐘圖',
            hourly: '小時圖',
            daily: '日線圖',
            weekly: '週線圖',
            monthly: '月線圖',
            quarterly: '季線圖'
        },
        period: {
            p1d: "1 日",
            p5d: "5 日",
            p1m: "1個月",
            p3m: "3個月",
            p6m: "6個月",
            p1y: "1 年",
            p2y: "2 年",
            p5y: "5 年",
            p10y: "10 年",
            ytd: "本年至今"
        },
        overview: {
        	pc:'上日收市'
        },
        currency: {
            'HKD': '$',
            'CNY': 'RMB',
            'USD': 'US$',
            'EUR': 'EUR'
        },
        product_type: {
            'EQTY': '股本證券',
            'CBBC': '牛熊證',
            'DW': '衍生權證',
            'ETP': '交易所買賣產品',
            'REIT': '房地產投資信託基金',
            'BOND': '債務證券'
        },
        status: {
            'new': '新上市',
            'ipo': 'IPO',
            'suspended': '停牌',
            'etp_new': '新'
        },
        filters: {
            'all': '全部'
        },
        hsirow:{
            hsi:'HSI FUTURES',
            hi:'high',
            lo:'low',
            vo:'volume'
        },
        future:{
            ps:'Product Spec',
            ete:'Export to Excel',
            texrow:'FUTURES',
            ds:'DAY SESSION',
            ns:'NIGHT SESSION',
            fu_ln:{
                'con':'Contract',
                'ls':'Last Traded',
                'nc':'Net Change',
                'se':'Prev.Day Settlement Price',
                'bd':'Bid',
                'as':'Ask',
                'op':'Open',
                'hi':'High',
                'lo':'Low',
                'vo':'Volume',
                'oi':'Prev.Day Open Interest'
            }
        },
        op:{
            title:'OPTIONS',
            msmmon:'MONTH',
            msmstr:'STRIKE',
            fr:'From',
            to:'To',
            txt:'IV Assumptions: [ Interest rate is 0.15% and dividend yield is 4.24%, per year.]',
            menutitle:{
                'cal':'CALL',
                'put':'PUT'
            },
            ln:{
                'oi':'OI',
                'vo':'Volume',
                'iv':'IV',
                'bd_as':'Bid/Ask',
                'ls':'Last',
                'stk':'Strike'
            }
        },
        sc:{
            sctitle:'Standard Combinations',
            tmctitle:'Tailor Made Combinations',
            ln:{
                'con':'Contract',
                'ty':'Type',
                'le':'Legs',
                'lt':'Last Trade',
                'bd':'Bid',
                'as':'Ask',
                'hi':'High',
                'lo':'Low',
                'vo':'Volume'
            }
        },
        loadrow:{
            loadmore:'更多',
            lastupdate:'Last Updated: ',
            items: '個項目',
            all: '全部'
        }
    }

};


// Resource - zh_HK
LabCI.WP.QuoteequitiesPageObj.PAGEOBJ_RESOURCEBUNDLE.zh_HK = {
    msg: {
        load: '更多',
        lastupdate: '更新: ',
        seeAll: '查看全部',
        'listpane': {
            'prevClose': '上日收市*',
            'open': '開市**',
            'turnover': '成交金額',
            'volume': '成交數量',
            'mktCap': '市值',
            'lotSize': '買賣單位',
            'bid': '買入價',
            'ask': '賣出價',
            'eps': '每股盈利',
            'pe': '市盈率',
            'divYield': '收益率',
            'intraday': '即日',
            'quote52': '52周',
            'high': '最高價',
            'low': '最低價',
            'openDesc': '** 開盤價是該日的首個交易價。開盤價於香港時間09:20:00 後匯報。',
            'prevcloseDesc': '* 上日收市價是最近期非零收市價或結算價格。',
            'delayDesc': '資訊於開市後提供並延時最少十五分鐘。',
            'rs_txt_1': '請',
            'rs_txt_a': '按此',
            'rs_txt_2': '查看附加於公司名稱後的RS標記之解釋',
            'h_share': '*** 只包括H股'
        },
        'related_level': {
            'title': '相關產品',
            'tabDWs': '衍生權證',
            'tabCBBCs': '牛熊證',
            'tabOptions': '股票期權',
            'tabFutures': '股票期貨',
            'code': '代號',
            'callput': '認購 / 認沽',
            'expiry': '到期日',
            'strike': '行使價 ',
            'lastPrc': '最後成交價',
            'bullbear': '牛證 / 熊證',
            'volume': '成交數量 ',
            'contract': '合約',
            'lastTrade': '最後成交',
            'oi': '未平倉合約',
            'noProduct': '沒有相關產品'
        },
        'detailpane': {
            'inputDesc': '股票搜尋',
            'exportExcel': '匯出到Excel格式',
            'chartDesc': '所有圖表均採用最後成交價或收市價。'
        },
        'company_level': {
            'title': '公司簡介',
            'issuedShares': '已發行股份*****',
            'industry': '行業',
            'listingDate': '上市日期',
            'finYrEnds': '財政年度結算日期',
            'chairman': '主席',
            'office': '總辦事處',
            'placeOfIncorp': '註冊地點',
            'listingCat': '上市類型',
            'reg': '過戶處',
            'listing_cat_txt_1': '* ',
            'listing_cat_txt_a': '查看',
            'listing_cat_txt_2': '公司資料報表',
            'listing_date_txt': '** 創業板轉至主板上市日期: ',
            'hsic_label': ' (恒生行業分類***)',
            'csic_label': ' (中證行業分類****)',
            'hsic_txt_1': '*** 恆生行業分類由恒生指數有限公司提供，請查看',
            'hsic_txt_link': '免責聲明及重要提示',
            'hsic_txt_2': '。',
            'csic_txt_1': '**** 中證行業分類由中證指數有限公司提供，並只適用於香港及上海兩地同時上市之公司。請查看',
            'csic_a_1': '免責聲明及重要提示',
            'csic_txt_2': '，並',
            'csic_a_2': '按此查看',
            'csic_txt_3': '所有滬港兩地同時上市公司名錄及其行業分類。',
            'issued_shares_txt_1': '***** 已發行股份的註釋',
            'issued_shares_txt_a': '按此',
            'issued_shares_txt_2': '。',
            'adjusted_no_txt_1': '****** 經調整股數的註釋',
            'adjusted_no_txt_a': '按此',
            'adjusted_no_txt_2': '。',
            'hsic_modal_header_1': '免責聲明',
            'hsic_modal_txt_1': '此部份提及的恒生行業分類系統的行業及業務類別分類資料﹝「此等資料」﹞由恒生指數有限公司提供。恒生指數有限公司及其任何控股公司及附屬公司及香港交易所資訊服務有限公司及其任何控股公司及附屬公司並無就此等資訊﹝包括但不限於此等資料是否適用或適合作任何用途，或任何人士使用或依賴此等資料會否獲取任何指定結果﹞提供任何形式保證，或就此等資料的準確性、完整性、及時性及／或一致性或就任何人士以任何方式遭受或招致的任何損失或損害承擔任何責任。任何人士一經向香港交易所資訊服務有限公司或其任何控股公司或附屬公司獲取此等資料或一經使用此等資料，即不得撤銷地及無條件地接受及同意受此免責聲明約束。',
            'hsic_modal_header_2': '重要提示 - 有關發布恒生行業分類系統資料',
            'hsic_modal_txt_2': '任何人士在發布全部或部份恒生行業分類系統的行業及業務類別資料予其他人士前，請聯絡「恒生指數有限公司」並須與其簽訂使用權協議。',
            'csic_modal_header': '免責聲明',
            'csic_modal_txt': '該等資料不包含任何明示或暗示的保證，包括對其適銷性、適銷品質、所有權、特定目的的適用性、安全性、準確性及非侵權等的保證。中證指數有限公司盡力確保其提供之資料準確可靠，但並不擔保該等資料的準確性和可靠性，且對因資料不准確、缺失或因依賴該等信息而造成的任何損失和損害不承擔任何責任〈無論其為侵權、違約或其他責任〉。',
            'is_modal_header': '發行股數/信託單位 - 備註',
            'is_modal_txt_1': '1. 聯交所已修訂《上市規則》，要求上市發行人在每個曆月結束後的第五個營業日上午8時30分前呈交「月報表」，定期更新上市發行人股本變動的資料；上市發行人亦須就已發行股本變動向聯交所呈交「翌日披露報表」：部分情況須在下一個營業日上午8時30分前呈交，而其他情況的呈交時間，則視乎是否觸及5%的最低界線水平及若干其他準則（如合併計算）。詳情請參閱《上市規則》。',
            'is_modal_txt_2': '2. 如欲取得上市發行人最近呈報的發行股數，請到香港交易所披露易網站（http://www.hkexnews.hk/index_c.htm）查閱上市發行人呈交的「月報表」及「翌日披露報表」。',
            'is_modal_txt_3': '3. 大部分情況下，此發行股數和上市發行人最近呈報的相符。如有差異，可能會是由於以下其中一個或多個的原因:',
            'is_modal_txt_3_1_h': '更新資料的時差',
            'is_modal_txt_3_1': '此發行股數通常在上市發行人呈交「月報表」或「翌日披露報表」後三天內更新。',
            'is_modal_txt_3_2_h': '供股或紅股',
            'is_modal_txt_3_2': '如上市發行人宣布供股或派發紅股，此發行股數將於除淨日當天調整。 ',
            'is_modal_txt_3_3_h': '股份拆細或合併',
            'is_modal_txt_3_3': '股份拆細或合併時，此發行股數將於生效日當天調整。',
            'is_modal_txt_3_4_h': '可換股票據轉換股份',
            'is_modal_txt_3_4': '如股本變動是由於可換股票據轉換成股份，此發行股數或會根據有關上市公司公告更新。',
            'is_modal_txt_4': '4. H股公司的發行股數只包含H股部分。',
            'is_modal_txt_5': '5. 預託證券的發行股數只包含已批准上市的香港預託證券數目。',
            'is_modal_txt_6': '6. 香港交易所依據此發行股數來計算市場市價總值。',
            'is_modal_txt_7': '7. 雙幣股票的發行股數為港幣股票及人民幣股票之總和。',
            'is_modal_txt_8': '8. 合訂證券的發行股數（包括普通股及優先股）指已獲准上市的合訂證券數目。',
            'is_modal_txt_9': '9. 合訂證券的發行信託單位指已獲准上市的信託單位數目。',
            'primary_market_label': '主要市場',
            'depositary_label': '存管人',
            'drratio_label': '預託證券比率',
            'ew_desc': '摘要',
            'ew_amt_os': '已發行額',
            'ew_sub_right': '每單位認購權',
            'ew_strike': '認購價/行使價',
            'ew_listdate': '上市日期',
            'ew_entitlement_ratio': '換股比率******',
            'ew_sub_per': '認購日期',
            'ew_underlying_code': '正股之股份代號',
            'ew_trd_ccy': '交易貨幣',
            'ew_to': ' 至 ',
            'ew_footer': '****** 換股比率一般代表每兌換一股或一單位相關資產所需的權證數目（可作出任何必要的調整，以反映任何資本化、供股、分配或類似情況）。'
         },
        'dividend_level': {
            'title': '股息',
            'dateAnnounced': '宣佈日期',
            'exDate': '除淨日',
            'details': '詳細資料',
            'finYrEnd': '財政年度結算日期',
            'bookClsDate': '截止過戶日期',
            'paymentDate': '派息日期*',
            'payDateDesc': '* 派息日期僅供說明之用。'
        },
        'comann_level': {
            'title': '公司公告',
            'disclaimer': '* 只提供過去三個月最多五個的公司公告，並已延遲的方式顯示。點擊「查看全部」連接，可在披露易網站查看更多公司公告。'
        }
    },
    link: {
        'registrar': 'http://www.hkex.com.hk/chi/stat/smstat/mthbull/rpt_registrars_c.htm',
        'rs_indicator': '/-/media/HKEX-Market/Listing/Rules-and-Guidance/Other-Resources/Listing-of-Overseas-Companies/Understanding-the-Risks-of-Investing-in-Overseas-Issuers/Regulation_S_e.pdf',
        'csic_doc': 'http://www.hkex.com.hk/chi/Invest/misc/documents/ic_for_ah_shares_tc.pdf'
    }
};
*/
