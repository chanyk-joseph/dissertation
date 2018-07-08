package tradingview_handlers

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"

	CLIUtils "github.com/chanyk-joseph/dissertation/stock/data-retriever-cli/utils"
	"github.com/chanyk-joseph/dissertation/stock/data-retriever/common/utils"
	"github.com/labstack/echo"
)

type StudyTemplateDataObj struct {
	Name    string `json:"name"`
	Content string `json:"content"`
}

func getStudyTemplatesStorageDirectory() string {
	storageDir, _ := filepath.Abs(filepath.Join(filepath.Dir(os.Args[0]), "study_templates_storage"))
	return storageDir
}

func GetStudyTemplatesHandler(c echo.Context) error {
	templateName := c.FormValue("template")

	targetFolder := getStudyTemplatesStorageDirectory()
	if templateName == "" {
		listResp := struct {
			Status string                 `json:"status"`
			Data   []StudyTemplateDataObj `json:"data"`
		}{}
		listResp.Status = "ok"
		listResp.Data = []StudyTemplateDataObj{}
		if !CLIUtils.HasFolder(targetFolder) {
			return c.JSON(200, listResp)
		}

		jsonFiles, _ := filepath.Glob(filepath.Join(targetFolder, "*.json"))
		for _, jsonFile := range jsonFiles {
			jsonContent, _ := ioutil.ReadFile(jsonFile)
			var data StudyTemplateDataObj
			json.Unmarshal(jsonContent, &data) //ignore error, To be improved
			listResp.Data = append(listResp.Data, data)
		}
		return c.JSON(200, listResp)
	}

	type ObjResp struct {
		Status string               `json:"status"`
		Data   StudyTemplateDataObj `json:"data"`
	}
	var data StudyTemplateDataObj
	jsonContent, err := ioutil.ReadFile(filepath.Join(targetFolder, templateName+".json"))
	if err != nil {
		return c.JSON(500, ObjResp{Status: "error"})
	}
	json.Unmarshal(jsonContent, &data) //ignore error, To be improved
	r := ObjResp{}
	r.Status = "ok"
	r.Data = data
	return c.JSON(200, r)
}

func PostStudyTemplatesHandler(c echo.Context) error {
	type Response struct {
		Status string `json:"status"`
	}

	targetFolder := getStudyTemplatesStorageDirectory()
	CLIUtils.CreateFolderIfNotExist(targetFolder)

	data := StudyTemplateDataObj{}

	//Save as chart
	jsonFilePath := filepath.Join(targetFolder, c.FormValue("name")+".json")
	os.Remove(jsonFilePath)
	data.Name = c.FormValue("name")
	data.Content = c.FormValue("content")

	err := ioutil.WriteFile(jsonFilePath, []byte(utils.ObjectToJSONString(data)), 0777)
	if err != nil {
		return c.JSON(500, Response{"error"})
	}
	return c.JSON(200, Response{"ok"})
}

func DeleteStudyTemplatesHandler(c echo.Context) error {
	templateName := c.FormValue("template")

	type Response struct {
		Status string `json:"status"`
	}

	if templateName != "" {
		targetFolder := getStudyTemplatesStorageDirectory()
		CLIUtils.CreateFolderIfNotExist(targetFolder)
		jsonFilePath := filepath.Join(targetFolder, templateName+".json")
		os.Remove(jsonFilePath)
		return c.JSON(200, Response{"ok"})
	}
	return c.JSON(404, Response{"error"})
}
