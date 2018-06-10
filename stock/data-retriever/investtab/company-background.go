package investtab

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

type Director struct {
	Name  string `json:"name"`
	Title string `json:"title"`
}

type ShareHolder struct {
	Name       string `json:"name"`
	Percentage string `json:"percentage"`
}

type CompanyBackground struct {
	Symbol           string        `json:"symbol"`
	UpdateTime       string        `json:"update_time"`
	RawData          string        `json:"raw_data"`
	FullName         *string       `json:"fullname"`
	ShortName        *string       `json:"shortname"`
	Chairman         *string       `json:"chairman"`
	Directors        []Director    `json:"directors"`
	CompanyEmail     *string       `json:"company_email"`
	Tel              *string       `json:"tel_no"`
	Fax              *string       `json:"fax_no"`
	CompanySecretary *string       `json:"company_secretary"`
	Solicitors       []string      `json:"solicitors"`
	ShareHolders     []ShareHolder `json:"share_holders"`
	BusinessPlace    *string       `json:"business_place"`
	Website          *string       `json:"inet_address"`
	Bankers          []string      `json:"bankers"`
	Auditors         *string       `json:"auditors"`
	ShareRegistrars  *string       `json:"share_registrars"`
	IncorpPlace      *string       `json:"incorp_place"`
}

func (companyBackground *CompanyBackground) ToString() string {
	buf, err := json.MarshalIndent(companyBackground, "", "	") //json.Marshal(companyBackground)
	if err != nil {
		panic(err)
	}

	return string(buf)
}

/*
func (companyBackground *CompanyBackground) UnmarshalJSON(data []byte) error {
	type Alias CompanyBackground
	aux := &struct {
		Directors    string `json:"directors"`
		ShareHolders string `json:"share_holders"`
		Bankers      string `json:"bankers"`
		*Alias
	}{
		Alias: (*Alias)(companyBackground),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	companyBackground.Directors = strings.Split(aux.Directors, "<br>")
	companyBackground.ShareHolders = strings.Split(aux.ShareHolders, "<br>")
	companyBackground.Bankers = strings.Split(aux.Bankers, "<br>")
	return nil
}
*/

func GetCompanyBackground(symbol string, useEng ...bool) (CompanyBackground, error) {
	url := fmt.Sprintf("https://api.investtab.com/api/quote/%s/company-info?locale=zh_hk", symbol)
	if len(useEng) > 0 && useEng[0] {
		url = fmt.Sprintf("https://api.investtab.com/api/quote/%s/company-info?locale=en", symbol)
	}

	var nullStringIfEmpty func(*string) *string
	nullStringIfEmpty = func(field *string) *string {
		if *field == "" {
			return nil
		}
		return field
	}
	var nullStringArrIfEmpty func([]string) []string
	nullStringArrIfEmpty = func(field []string) []string {
		if len(field) == 1 && (field)[0] == "" {
			return nil
		}
		return field
	}

	fmt.Println("Trying: " + symbol)
	response, _ := http.Get(url)
	var companyBackground CompanyBackground
	companyBackground.Symbol = symbol
	companyBackground.UpdateTime = time.Now().UTC().String()
	if response.StatusCode == 200 && response.ContentLength > 0 {
		buf, _ := ioutil.ReadAll(response.Body)

		type Alias CompanyBackground
		aux := &struct {
			Directors    string `json:"directors"`
			ShareHolders string `json:"share_holders"`
			Bankers      string `json:"bankers"`
			Solicitors   string `json:"solicitors"`
			*Alias
		}{
			Alias: (*Alias)(&companyBackground),
		}
		err := json.Unmarshal(buf, &aux)
		companyBackground.Bankers = strings.Split(aux.Bankers, "<br>")
		companyBackground.Solicitors = strings.Split(aux.Solicitors, "<br>")

		var tempArr []string
		tempArr = strings.Split(aux.Directors, "<br>")
		if len(tempArr) >= 1 && tempArr[0] != "" {
			for _, director := range tempArr {
				var re = regexp.MustCompile(`(.*) \((.*?)\)`)
				dirObj := Director{
					Name:  re.FindStringSubmatch(director)[1],
					Title: re.FindStringSubmatch(director)[2],
				}
				companyBackground.Directors = append(companyBackground.Directors, dirObj)
			}
		}
		tempArr = strings.Split(aux.ShareHolders, "<br>")
		if len(tempArr) >= 1 && tempArr[0] != "" {
			for _, shareHolder := range tempArr {
				var re = regexp.MustCompile(`(.*) \((.*?)%\)`)
				shareHolderObj := ShareHolder{
					Name:       re.FindStringSubmatch(shareHolder)[1],
					Percentage: re.FindStringSubmatch(shareHolder)[2],
				}
				companyBackground.ShareHolders = append(companyBackground.ShareHolders, shareHolderObj)
			}
		}

		companyBackground.Bankers = nullStringArrIfEmpty(companyBackground.Bankers)
		companyBackground.Solicitors = nullStringArrIfEmpty(companyBackground.Solicitors)
		companyBackground.Chairman = nullStringIfEmpty(companyBackground.Chairman)
		companyBackground.CompanySecretary = nullStringIfEmpty(companyBackground.CompanySecretary)
		companyBackground.Auditors = nullStringIfEmpty(companyBackground.Auditors)
		companyBackground.BusinessPlace = nullStringIfEmpty(companyBackground.BusinessPlace)
		companyBackground.Website = nullStringIfEmpty(companyBackground.Website)
		companyBackground.CompanyEmail = nullStringIfEmpty(companyBackground.CompanyEmail)
		companyBackground.Tel = nullStringIfEmpty(companyBackground.Tel)
		companyBackground.Fax = nullStringIfEmpty(companyBackground.Fax)

		companyBackground.RawData = string(buf)
		return companyBackground, err
	} else if response.StatusCode == 429 {
		buf, _ := ioutil.ReadAll(response.Body)

		re := regexp.MustCompile(`Expected available in ([0-9]+)\D`)
		waitSecStr := re.FindStringSubmatch(string(buf))[1]
		waitSec, _ := strconv.Atoi(waitSecStr)

		fmt.Println(symbol + " | " + string(buf) + " | Wait " + waitSecStr + "s")
		time.Sleep(time.Duration(waitSec) * time.Second)

		return GetCompanyBackground(symbol)
	} else {
		return companyBackground, nil
	}
}

func SortBySymbol(arr *[]CompanyBackground, ascending bool) {
	sort.Slice(*arr, func(i, j int) bool {
		s1 := (*arr)[i]
		s2 := (*arr)[j]

		re := regexp.MustCompile(`0*?([1-9]+[0-9]*?):HK`)
		id1, _ := strconv.Atoi((re.FindStringSubmatch(s1.Symbol))[1])
		id2, _ := strconv.Atoi((re.FindStringSubmatch(s2.Symbol))[1])

		if ascending {
			return id1 < id2
		}
		return id1 > id2
	})
}

func getAllCompanyBackgroundFromInvestTab() {
	total := 99999
	concurrency := 6
	guard := make(chan int, concurrency)
	throttle := time.Tick(time.Second / 6)
	finished := make(chan int, total)

	CompanyBackgrounds := make(chan CompanyBackground, total)
	for i := 1; i <= total; i++ {
		<-throttle
		guard <- i

		go func(i int) {
			symbol := fmt.Sprintf("%05d:HK", i)

			companyBackground, err := GetCompanyBackground(symbol)
			if err == nil {
				CompanyBackgrounds <- companyBackground
			}

			<-guard
			finished <- i
		}(i)
	}

	for i := 0; i < total; i++ {
		<-finished
	}
	fmt.Println("Completed")

	var temp []CompanyBackground
	fmt.Println("Before: " + strconv.Itoa(len(temp)) + " | " + strconv.Itoa(len(CompanyBackgrounds)))
	ciLen := len(CompanyBackgrounds)
	for i := 0; i < ciLen; i++ {
		ci := <-CompanyBackgrounds
		temp = append(temp, ci)
	}
	fmt.Println("After: " + strconv.Itoa(len(temp)) + " | " + strconv.Itoa(len(CompanyBackgrounds)))

	sort.Slice(temp, func(i, j int) bool {
		s1 := temp[i]
		s2 := temp[j]

		re := regexp.MustCompile(`0*?([1-9]+[0-9]*?):HK`)
		id1, _ := strconv.Atoi((re.FindStringSubmatch(s1.Symbol))[1])
		id2, _ := strconv.Atoi((re.FindStringSubmatch(s2.Symbol))[1])

		return id1 < id2
	})

	buf3, err := json.Marshal(temp)
	if err != nil {
		panic(err)
	}
	err = ioutil.WriteFile("test.json", buf3, 0644)
	if err != nil {
		panic(err)
	}
}

// // oldFormatToNew("C:/Users/Joseph/Desktop/playground/Stock_Info/test.json", "C:/Users/Joseph/Desktop/playground/Stock_Info/test_new.json")
// buf, e := ioutil.ReadFile("C:/Users/Joseph/Desktop/playground/Stock_Info/test_new.json")
// check(e)
// var arr []tasks.CompanyBackground
// err := json.Unmarshal(buf, &arr)
// check(err)

// var arr2 []tasks.CompanyBackground
// for _, obj := range arr {
// 	engObj, err := tasks.GetCompanyBackground(obj.Symbol, true)
// 	check(err)
// 	arr2 = append(arr2, engObj)
// }
// tasks.SortBySymbol(&arr2, true)
// fmt.Println(len(arr2))
// buf, err = json.MarshalIndent(arr2, "", "	")
// check(err)
// err = ioutil.WriteFile("C:/Users/Joseph/Desktop/playground/Stock_Info/test_new_en.json", buf, 0644)
// check(err)
