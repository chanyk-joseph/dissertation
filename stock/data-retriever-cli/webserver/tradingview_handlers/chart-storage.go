package tradingview_handlers

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
	"github.com/twinj/uuid"
)

type ChartDrawingDataObj struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Timestamp  int64  `json:"timestamp"`
	Symbol     string `json:"symbol"`
	Resolution string `json:"resolution"`
	Content    string `json:"content"`
}

func getChartDrawingStorageDirectory() string {
	storageDir, _ := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), "chart_storage"))
	return storageDir
}

func GetChartHandler(c echo.Context) error {
	// clientID := c.FormValue("client")
	// userID := c.FormValue("user")
	chartID := c.FormValue("chart")

	targetFolder := getChartDrawingStorageDirectory()
	if chartID == "" {
		listResp := struct {
			Status string                `json:"status"`
			Data   []ChartDrawingDataObj `json:"data"`
		}{}
		listResp.Status = "ok"
		listResp.Data = []ChartDrawingDataObj{}
		if !CLIUtils.HasFolder(targetFolder) {
			return c.JSON(200, listResp)
		}

		jsonFiles, _ := filepath.Glob(filepath.Join(targetFolder, "*.json"))
		for _, jsonFile := range jsonFiles {
			jsonContent, _ := ioutil.ReadFile(jsonFile)
			var data ChartDrawingDataObj
			json.Unmarshal(jsonContent, &data) //ignore error, To be improved
			listResp.Data = append(listResp.Data, data)
		}
		return c.JSON(200, listResp)
	}

	type ObjResp struct {
		Status string              `json:"status"`
		Data   ChartDrawingDataObj `json:"data"`
	}
	var data ChartDrawingDataObj
	jsonContent, err := ioutil.ReadFile(filepath.Join(targetFolder, chartID+".json"))
	if err != nil {
		return c.JSON(500, ObjResp{Status: "error"})
	}
	json.Unmarshal(jsonContent, &data) //ignore error, To be improved
	r := ObjResp{}
	r.Status = "ok"
	r.Data = data
	return c.JSON(200, r)
}

func PostChartHandler(c echo.Context) error {
	// clientID := c.FormValue("client")
	// userID := c.FormValue("user")
	chartID := c.FormValue("chart")

	type Response struct {
		Status string `json:"status"`
		ID     string `json:"id"`
	}

	targetFolder := getChartDrawingStorageDirectory()
	CLIUtils.CreateFolderIfNotExist(targetFolder)

	data := ChartDrawingDataObj{}

	//Save Chart
	if chartID == "" {
		data.ID = uuid.NewV4().String()
		data.Timestamp = time.Now().UTC().Unix()
		data.Name = c.FormValue("name")
		data.Resolution = c.FormValue("resolution")
		data.Symbol = c.FormValue("symbol")
		data.Content = c.FormValue("content")

		jsonFilePath := filepath.Join(targetFolder, data.ID+".json")
		os.Remove(jsonFilePath)
		err := ioutil.WriteFile(jsonFilePath, []byte(utils.ObjectToJSONString(data)), 0777)
		if err != nil {
			return c.JSON(500, Response{"error", data.ID})
		}
		return c.JSON(200, Response{"ok", data.ID})
	}

	//Save as chart
	jsonFilePath := filepath.Join(targetFolder, chartID+".json")
	os.Remove(jsonFilePath)
	data.ID = chartID
	data.Timestamp = time.Now().UTC().Unix()
	data.Name = c.FormValue("name")
	data.Resolution = c.FormValue("resolution")
	data.Symbol = c.FormValue("symbol")
	data.Content = c.FormValue("content")

	err := ioutil.WriteFile(jsonFilePath, []byte(utils.ObjectToJSONString(data)), 0777)
	if err != nil {
		return c.JSON(500, Response{"error", data.ID})
	}
	return c.JSON(200, Response{"ok", data.ID})
}

func DeleteChartHandler(c echo.Context) error {
	// clientID := c.FormValue("client")
	// userID := c.FormValue("user")
	chartID := c.FormValue("chart")

	type Response struct {
		Status string `json:"status"`
	}

	if chartID != "" {
		targetFolder := getChartDrawingStorageDirectory()
		CLIUtils.CreateFolderIfNotExist(targetFolder)
		jsonFilePath := filepath.Join(targetFolder, chartID+".json")
		os.Remove(jsonFilePath)
		return c.JSON(200, Response{"ok"})
	}
	return c.JSON(404, Response{"error"})
}
