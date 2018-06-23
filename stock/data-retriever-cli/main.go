package main

func main() {
	webserver := SetupWebserver()
	webserver.Start(":8888")
}
