__customIndicators = [
    {
        // Replace the <study name> with your study name
        // The name will be used internally by the Charting Library
        name: "joseph-indicator",
        metainfo: {
            "_metainfoVersion": 40,
            "id": "joseph-indicator@tv-basicstudies-1",
            "scriptIdPart": "",
            "name": "joseph-indicator",
    
            // This description will be displayed in the Indicators window
            // It is also used as a "name" argument when calling the createStudy method
            "description": "joseph-indicator",
    
            // This description will be displayed on the chart
            "shortDescription": "joseph-indicator",
    
            "is_hidden_study": true,
            "is_price_study": true,
            "isCustomIndicator": true,

            "plots": [{"id": "plot_0", "type": "line"}],
            "defaults": {
                "styles": {
                    "plot_0": {
                        "linestyle": 0,
                        "visible": true,

                        // Make the line thinner
                        "linewidth": 1,

                        // Plot type is Line
                        "plottype": 2,

                        // Show price line
                        "trackPrice": true,

                        "transparency": 40,

                        // Set the plotted line color to dark red
                        "color": "#880000"
                    }
                },

                // Precision is set to one digit, e.g. 777.7
                "precision": 1,

                "inputs": {}
            },
            "styles": {
                "plot_0": {
                    // Output name will be displayed in the Style window
                    "title": "Equity value",
                    "histogramBase": 0,
                }
            },
            "inputs": [],
        },
    
        constructor: function() {
            this.init = function(context, inputCallback) {
                this._context = context;
                this._input = inputCallback;
    
                // Define the symbol to be plotted.
                // Symbol should be a string.
                // You can use PineJS.Std.ticker(this._context) to get the selected symbol's ticker.
                // For example,
                //    var symbol = "AAPL";
                //    var symbol = "#EQUITY";
                //    var symbol = PineJS.Std.ticker(this._context) + "#TEST";
                var symbol = "joseph-indicator"; //"00002.HK";
                this._context.new_sym(symbol, PineJS.Std.period(this._context), PineJS.Std.period(this._context));
            };
    
            this.main = function(context, inputCallback) {
                this._context = context;
                this._input = inputCallback;
    
                this._context.select_sym(1);
    
                // You can use following built-in functions in PineJS.Std object:
                //    open, high, low, close
                //    hl2, hlc3, ohlc4
                var v = PineJS.Std.close(this._context);
                return [v];
            }
        }
    }
];