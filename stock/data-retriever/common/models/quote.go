package models

type StandardQuote struct {
	Open   float64 `json:"open"`
	Low    float64 `json:"low"`
	High   float64 `json:"high"`
	Close  float64 `json:"close"`
	Volume int64   `json:"volume"`
}

type StandardQuoteFromAllProviders struct {
	StandardSymbol
	Quotes map[string]StandardQuote `json:"quotes"`
}

type RawQuoteFromAllProviders struct {
	StandardSymbol
	Quotes map[string]interface{} `json:"quotes"`
}

type QuotesOfHSIComponents struct {
	Quotes []StandardQuoteFromAllProviders `json:"quotes"`
}
