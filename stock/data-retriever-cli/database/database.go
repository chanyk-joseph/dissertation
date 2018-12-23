package database

import (
	"database/sql"
	"fmt"
)

var dbAddr string
var dbUser string
var dbPW string

func SetupDatabase(addr string, username string, password string) {
	dbAddr = addr
	dbUser = username
	dbPW = password
}

func GetDB(dbName string) (db *sql.DB, err error) {
	connectStr := fmt.Sprintf("%s:%s@tcp(%s)/%s", dbUser, dbPW, dbAddr, dbName)
	return sql.Open("mysql", connectStr)
}

func Query(sql string) (result []map[string]string, err error) {
	result = []map[string]string{}

	fmt.Println(sql)
	db, err := GetDB("dissertation")
	if err != nil {
		return
	}
	defer db.Close()

	rows, err := db.Query(sql)
	if err != nil {
		return
	}
	cols, err := rows.Columns()
	if err != nil {
		return
	}
	for rows.Next() {
		// Create a slice of interface{}'s to represent each column,
		// and a second slice to contain pointers to each item in the columns slice.
		columns := make([]interface{}, len(cols))
		columnPointers := make([]interface{}, len(cols))
		for i, _ := range columns {
			columnPointers[i] = &columns[i]
		}

		// Scan the result into the column pointers...
		if err := rows.Scan(columnPointers...); err != nil {
			return nil, err
		}

		// Create our map, and retrieve the value for each column from the pointers slice,
		// storing it in the map with the name of the column as the key.
		m := make(map[string]string)
		for i, colName := range cols {
			val := columnPointers[i].(*interface{})
			m[colName] = string((*val).([]uint8))
		}

		result = append(result, m)
	}
	return
}
