#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>

#include "ConfigFile.h"
#include "MarketPriceRetriever.h"

#include <unistd.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <fstream>
#include <sstream>

string getCurrentSystemTime(){
    char fmt[64];
    char buf[64];
    struct timeval tv;
    struct tm *tm;

    gettimeofday (&tv, NULL);
    tm = localtime (&tv.tv_sec);
    strftime (fmt, sizeof (fmt), "%Y/%m/%d %H:%M:%S:%%06u", tm);
    snprintf (buf, sizeof (buf), fmt, tv.tv_usec);

    return string(buf);
}

// SPApiProduct.ProdType is empty for demo account
// need to judge the product type in an alternative way
int getProductType(SPApiProduct p){
	if (string(1, p.CallPut) != "-"){
		return 2;	//options
	}else if( string(p.ProdCode).find("/") != std::string::npos ){
		return 3;	//spreads
	}else{
		return 1;	//futures
	}
}

MarketPriceRetriever::MarketPriceRetriever(string apiEndPoint, int apiPort, string apiLicense, string apiID, string userID, string password){
    this->apiEndPoint   = apiEndPoint;
    this->apiPort       = apiPort;
    this->apiLicense    = apiLicense;
    this->apiID         = apiID;
    this->userID        = userID;
    this->password      = password;
}
MarketPriceRetriever::MarketPriceRetriever(string configFilePath){
	cout << "Config file path: " << configFilePath << endl;

	ConfigFile cf(configFilePath);
	string apiEndPoint	= cf.Value("Login", "host");
	double apiPort		= cf.Value("Login", "port");
	string apiLicense	= cf.Value("Login", "app_license");
	string apiID		= cf.Value("Login", "app_id");
	string userID		= cf.Value("Login", "username");
	string password		= cf.Value("Login", "password");
	string outFilePrefix= cf.Value("Path", "marketdata_filename_prefix");

    this->apiEndPoint   = apiEndPoint;
    this->apiPort       = apiPort;
    this->apiLicense    = apiLicense;
    this->apiID         = apiID;
    this->userID        = userID;
    this->password      = password;

	this->outputMarketDataFileNamePrefix = outFilePrefix;

	map<string, string> marketCode2IdMap;
	for(int i=1; i<=100; i++){
		string id = to_string(i);
		try{
			string marketCode = cf.Value("Markets", id);
			marketCode2IdMap[marketCode] = id;
		}catch(...){
			break;
		}
	}

	map<string, vector<string>> productsOfMarket;
	map<string, string> instCode2MarketIdMap;
	for (auto const& x : marketCode2IdMap){
		string marketId = x.second;
		try{
			string products = cf.Value("Products", marketId);
			vector<std::string> tmp;
			boost::split(tmp, products, boost::is_any_of(","));
			productsOfMarket[marketId] = tmp;

			for(int i=0; i<tmp.size(); i++)
				instCode2MarketIdMap[tmp.at(i)] = marketId;

			// cout << marketId << endl;
			// for(int i=0; i<tmp.size(); i++){
			// 	cout << tmp.at(i) << endl;
			// }
		}catch(...){
			continue;
		}
	}

	apiProxyWrapper.SPAPI_RegisterApiProxyWrapperReply(this);
	apiProxyWrapper.SPAPI_SetLanguageId(0);
	apiProxyWrapper.SPAPI_Initialize();
	apiProxyWrapper.SPAPI_SetLoginInfo(&apiEndPoint[0], apiPort, &apiLicense[0], &apiID[0], &userID[0], &password[0]);

	if( apiProxyWrapper.SPAPI_Login() != 0 ){
		return;
	}

	while(true){
		if( isAllServiceConnected() ){
			break;
		}
		usleep(50000);
	}
	if( !retrieveAllInstruments() )
		return;

	cout << "Retrieving Chosen Products List ..." << endl;
	for (auto const& x : marketCode2IdMap){
		string marketId = x.second;
		vector<string> products = productsOfMarket[marketId];

		string queryStr = boost::algorithm::join(products, ",");
		retrievingFlag = true;
		if (apiProxyWrapper.SPAPI_LoadProductInfoListByCode(&queryStr[0]) > 0 ){
			cout << queryStr << endl;
			while( retrievingFlag )
				usleep(500000);
		}
	}

	apiProxyWrapper.SPAPI_GetProduct(productsList);
	cout << "Loaded Products Count: " << productsList.size() << endl;

	for(int i=0; i<productsList.size(); i++){
		SPApiProduct p = productsList.at(i);
		string productType = to_string(getProductType(p));
		string marketId = instCode2MarketIdMap[p.InstCode];

		try{
			// If the market of the product is not listed in config, then it will not be subscribed
			string products = cf.Value("ProductTypes", marketId);
			vector<std::string> pType;
			boost::split(pType, products, boost::is_any_of(","));

			vector<string>::iterator it = find(pType.begin(), pType.end(), productType);
			if (it != pType.end()) {
				// product type is in the config list, so we subscribe the product
				SubscribeProductPrice(p.ProdCode);
			}
		}catch(...){}
	}
}
MarketPriceRetriever::~MarketPriceRetriever(void){}

void MarketPriceRetriever::Unload(){
	int rc = apiProxyWrapper.SPAPI_Uninitialize(&userID[0]);

	if (rc == 0)		printf("\nUninitialize DLL OK\n");
	else if (rc == -1)	printf("\nUsers Not Logout\n");
	else				printf("\nUninitialize DLL Catch\n");
}

bool MarketPriceRetriever::Init(string outputCSV_Path){
    cout << "Logging In: " << userID << endl;
	cout << outputCSV_Path << endl;
	apiProxyWrapper.SPAPI_RegisterApiProxyWrapperReply(this);
	apiProxyWrapper.SPAPI_SetLanguageId(0);
	apiProxyWrapper.SPAPI_Initialize();
	apiProxyWrapper.SPAPI_SetLoginInfo(&apiEndPoint[0], apiPort, &apiLicense[0], &apiID[0], &userID[0], &password[0]);

	if( apiProxyWrapper.SPAPI_Login() != 0 ){
		return false;
	}

	while(true){
		if( isAllServiceConnected() ){
			break;
		}
		usleep(50000);
	}

	if( !retrieveAllProducts() )
		return false;

	cout << "Init Completed" << endl;
	return true;
}

void MarketPriceRetriever::InitOutputCSVForProduct(string prodCode){
	string priceCSVPath = getOutputPath(prodCode, false);
	string tickerCSVPath = getOutputPath(prodCode, true);

	prodCodePriceOutputFileStreams[prodCode] = new ofstream(priceCSVPath, std::ios_base::app);
	prodCodeTickerOutputFileStreams[prodCode] = new ofstream(tickerCSVPath, std::ios_base::app);

	ifstream priceCSVFile(priceCSVPath);
	if( priceCSVFile.peek() == std::ifstream::traits_type::eof() ){
		// if the file is empty, then append csv headers
		(*prodCodePriceOutputFileStreams[prodCode]) << "SysTime,yyyymmdd_HHMMSS_f,Product Code,Last Trade Price,Cummulated Last Traded Volume,Buy Delimiter,First Bid Price,First Bid Volume,Sec Bid Price,Sec Bid Volume,Third Bid Price,Third Bid Volume,Fourth Bid Price,Fourth Bid Volume,Fifth Bid Price,Fifth bid volume,Ask Price Delimiter,First Ask Price,First Ask Volume,Sec Ask Price,Sec Ask Volume,Third Ask Price,Third Ask Volume,Fourth Ask Price,Fourth Ask Volume,Fifth Ask Price,Fifth Ask volume" << endl;
	}

	ifstream tickerCSVFile(tickerCSVPath);
	if( tickerCSVFile.peek() == std::ifstream::traits_type::eof() ){
		// if the file is empty, then append csv headers
		(*prodCodeTickerOutputFileStreams[prodCode]) << "SysTime,yyyymmdd_HHMMSS_f,Product Code,Price,Qty,DealSrc" << endl;
	}
}

bool MarketPriceRetriever::retrieveAllInstruments(){
	cout << "Retrieving Instruments List..." << endl;
	int rc = apiProxyWrapper.SPAPI_LoadInstrumentList();
	if( rc > 0 ){
		while(!isInstrumentsListDownloaded){
			usleep(1000000);
		}
	}
	if (apiProxyWrapper.SPAPI_GetInstrument(instrumentsList) != 0)
		return false;
	if (instrumentsList.size() == 0){
		printf("\n No Instrument \n");
		return false;
	}

	// for(int i=0; i<instrumentsList.size(); i++){
	// 	SPApiInstrument &item = instrumentsList.at(i);
		// if( item.MarketCode == "HKEX" )
	// 		cout << item.InstCode << " | inst name: " << item.InstName << " | instType: " << item.InstType << " | market code: " << item.MarketCode << endl;
	// }

	cout << "Loaded Instruments Count: " << instrumentsList.size() << endl;
	return true;
}

bool MarketPriceRetriever::retrieveAllProducts(){
	if( !retrieveAllInstruments() )
		return false;

	cout << "Retrieving Products List..." << endl;
	for ( int i = 0; i < instrumentsList.size(); ){
		int begin	= i;
		int end		= i + maxConcurrencyOfProductRetrieve - 1;
		if( end >= instrumentsList.size() )
			end = instrumentsList.size() - 1;

		vector<string> tmp;
		for (int j=begin; j<=end; j++){
			SPApiInstrument& instrument = instrumentsList[j];
			tmp.push_back(instrument.InstCode);
		}
		string queryStr = boost::algorithm::join(tmp, ",");
		retrievingFlag = true;
		if (apiProxyWrapper.SPAPI_LoadProductInfoListByCode(&queryStr[0]) > 0 ){
			cout << queryStr << endl;
			while( retrievingFlag )
				usleep(500000);
		}

		i = end + 1;
	}

	apiProxyWrapper.SPAPI_GetProduct(productsList);
	cout << "Loaded Products Count: " << productsList.size() << endl;
}

bool MarketPriceRetriever::SubscribeProductPrice(string prodCode){
	cout << "Subscribe Product Price: " << prodCode << endl;

	InitOutputCSVForProduct(prodCode);

	//Check if the product is already subscribed
	for( int i=0; i<subscribedProductsList.size(); i++ ){
		if( subscribedProductsList.at(i) == prodCode )
			return true;
	}
	
	if (apiProxyWrapper.SPAPI_SubscribePrice(&userID[0], &prodCode[0], 1) != 0){
		return false;
	}
	if (apiProxyWrapper.SPAPI_SubscribeTicker(&userID[0], &prodCode[0], 1) != 0){
		return false;
	}

	subscribedProductsList.push_back(prodCode);
	return true;
}

bool MarketPriceRetriever::UnsubsribteProductPrice(string prodCode){
	bool found = false;
	for( int i=0; i<subscribedProductsList.size(); i++ ){
		if( subscribedProductsList.at(i) == prodCode )
			found = true;
	}
	if( !found )
		return true;

}

void MarketPriceRetriever::OnApiPriceUpdate(const SPApiPrice *price){
	struct tm *tblock;
	if (price == NULL)
		return;

	/*string bidQ = CommonUtils::GetBigQtyStr(price->BidQty[0], true);
	string bidPrice = CommonUtils::BidAskPriceStr(price->Bid[0], price->DecInPrice);
	string askPrice = CommonUtils::BidAskPriceStr(price->Ask[0], price->DecInPrice);
	string askQ = CommonUtils::GetBigQtyStr(price->AskQty[0], true);*/
    time_t TheTime = price->Timestamp;
    tblock = localtime(&TheTime);
	// cout <<"Price:"+ string(price->ProdCode)<< '\t' << price->BidQty[0] << '\t' << price->Bid[0] << '\t' << price->Ask[0] << '\t' << price->AskQty[0]<< '\t' << price->Timestamp<< "["<< tblock->tm_hour <<":"<< tblock->tm_min << ":" << tblock->tm_sec << "]" << endl;

	char tmpBuffer[256];
	strftime(tmpBuffer, sizeof(tmpBuffer), "%Y%m%d_%H%M%S_000000", tblock);
	string timeStr = string(tmpBuffer);

	stringstream ss;
	ss	<< timeStr << "," 
		<< string(price->ProdCode) << ","
		<< price->Last[0] << ","				// Last Trade Price
		<< price->TurnoverVol << ",";			// Cummulated Last Traded Volume
	ss	<< "B" << ",";							// Buy Delimiter
	for(int i=0; i<5; i++){
		ss	<< price->Bid[i] << "," 
			<< price->BidQty[i] << ",";
	}
	ss	<< "A" << ",";							// Ask Price Delimiter
	for(int i=0; i<5; i++){
		ss	<< price->Ask[i] << "," 
			<< price->AskQty[i];

		if( i != 4 )
			ss << ",";
	}

	string temp = ss.str();
	string sysTime = getCurrentSystemTime();
	cout << "Sys Time: " << sysTime << " | " << temp << endl;
	*prodCodePriceOutputFileStreams[price->ProdCode] << sysTime << "," << temp << endl;
}


void MarketPriceRetriever::OnApiTickerUpdate(const SPApiTicker *ticker){
    struct tm *tblock;
    time_t TheTime = ticker->TickerTime;
    tblock = localtime(&TheTime);

	char tmpBuffer[256];
	strftime(tmpBuffer, sizeof(tmpBuffer), "%Y%m%d_%H%M%S_000000", tblock);
	string timeStr = string(tmpBuffer);

	string sysTime = getCurrentSystemTime();
	string content = timeStr + "," + ticker->ProdCode + "," + to_string(ticker->Price) + "," + to_string(ticker->Qty) + "," + to_string(ticker->DealSrc);

	cout << "Sys Time: " << sysTime << " | Ticker: "+ string(ticker->ProdCode)<< "\tPrice: " << ticker->Price << "\tQty: " << ticker->Qty << "\tTransaction Time: " << tblock->tm_hour <<":"<< tblock->tm_min << ":" << tblock->tm_sec << "\tDealSrc: " << ticker->DealSrc << endl;
	*prodCodeTickerOutputFileStreams[ticker->ProdCode] << sysTime << "," << content << endl;
}


void MarketPriceRetriever::OnProductListByCodeReply(long req_id, char *inst_code, bool is_ready, char *ret_msg){
	// printf("\nProductListByCodeReply[Request Id:%d](inst code:%s):%s. Ret Msg:%s\n",req_id, inst_code, is_ready?"Ok":"No", ret_msg);
	retrievingFlag = !is_ready;
}

void MarketPriceRetriever::OnInstrumentListReply(long req_id, bool is_ready, char *ret_msg){
	isInstrumentsListDownloaded = is_ready;
}

void MarketPriceRetriever::OnApiQuoteRequestReceived(char *product_code, char buy_sell, long qty){
	cout <<"Quote Request: ProductCode:"+ string(product_code) << "  b_s:"<< buy_sell << " qty="<< qty << endl;
     //(buy_sell == 0)  strcpy(bs, "Both");
	 //(buy_sell == 'B')strcpy(bs, "Buy");
	 //(buy_sell == 'S')strcpy(bs, "Sell");
}

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}
string MarketPriceRetriever::getOutputPath(string prodCode, bool isTicker){
	string tmpProdCode = prodCode;
	replaceAll(tmpProdCode, "/", "_");
	if ( isTicker )
		return outputMarketDataFileNamePrefix + "_" + tmpProdCode + "_ticker.csv";
	else
		return outputMarketDataFileNamePrefix + "_" + tmpProdCode + ".csv";
}




















char *OutputOrderStatus(char status){
	switch(status){
		case ORDSTAT_SENDING:		return "sending";
		case ORDSTAT_WORKING:		return "working";
		case ORDSTAT_INACTIVE:		return "inactive";
		case ORDSTAT_PENDING:		return "pending";
		case ORDSTAT_ADDING:		return "adding";
		case ORDSTAT_CHANGING:		return "changing";
		case ORDSTAT_DELETING:		return "deleting";
		case ORDSTAT_INACTING:		return "inacting";
		case ORDSTAT_PARTTRD_WRK:	return "parttrd_wrk";
		case ORDSTAT_TRADED:		return "traded";
		case ORDSTAT_DELETED:		return "deleted";
		case ORDSTAT_APPROVEWAIT:   return "approve wait";
		case ORDSTAT_TRADEDREP:		return "traded & reported";
		case ORDSTAT_DELETEDREP:	return "deleted & reported";
		case ORDSTAT_RESYNC_ABN:	return "resync abnormal";
		case ORDSTAT_PARTTRD_DEL:	return "partial traded & deleted";
		case ORDSTAT_PARTTRD_REP:	return "partial traded & reported (deleted)";
	}
	return "?????unknown order status?????";
}

char *StrToArr(string str){
    return &str[0];
}

bool MarketPriceRetriever::isAllServiceConnected(){
	for(int i=0; i<3; i++){
		if( !this->serviceConnectionStatus[i] )
			return false;
	}
	return true;
}

void MarketPriceRetriever::OnLoginReply(char *user_id, long ret_code,char *ret_msg){
	if (ret_code != 0) {
		cout << "Login ErrMsg: " + string(ret_msg) << endl;
	}
}

void MarketPriceRetriever::OnConnectedReply(long host_type, long con_status){
/*
host_type：返回发生改变的行情状态服务器的ID.
	80,81： 表示交易连接
	83：表示一般价格连接.
	88：表示一般资讯连接.
con_status：
	1：连接中，2：已连接，3：连接错误， 4：连接失败
*/
	switch (host_type) {
		case 80:
		case 81:
			cout << "Host type :["<< host_type <<"][" << con_status << "]Transaction... Please wait!"  << endl;
			this->serviceConnectionStatus[0] = (con_status==2)?true:false;
			break;
		case 83:
			cout << "Host type :["<< host_type <<"][" << con_status << "]Quote price port... Please wait"  << endl;
			this->serviceConnectionStatus[1] = (con_status==2)?true:false;
			break;
		case 88:
			cout << "Host type :["<< host_type <<"][" << con_status << "]Information Link... Please wait!"  << endl;
			this->serviceConnectionStatus[2] = (con_status==2)?true:false;
			break;
	}
}

void MarketPriceRetriever::OnApiOrderRequestFailed(tinyint action, const SPApiOrder *order, long err_code, char *err_msg){}

void MarketPriceRetriever::OnApiOrderReport(long rec_no, const SPApiOrder *order){}

void MarketPriceRetriever::OnApiOrderBeforeSendReport(const SPApiOrder *order){}

void MarketPriceRetriever::OnAccountLoginReply(char *accNo, long ret_code, char* ret_msg)
{
	cout << "Account Login Reply: acc_no="+ string(accNo) << " ret_code="<< ret_code << " ret_msg="+ string(ret_msg)  << endl;
}


void MarketPriceRetriever::OnAccountLogoutReply(char *accNo, long ret_code, char* ret_msg){}

void MarketPriceRetriever::OnAccountInfoPush(const SPApiAccInfo *acc_info)
{
	cout <<"AccInfo: acc_no="+ string(acc_info->ClientId)<< " AE="+ string(acc_info->AEId)<< " BaseCcy="+ string(acc_info->BaseCcy) << endl;
}

void MarketPriceRetriever::OnAccountPositionPush(const SPApiPos *pos){}

void MarketPriceRetriever::OnUpdatedAccountPositionPush(const SPApiPos *pos){}

void MarketPriceRetriever::OnUpdatedAccountBalancePush(const SPApiAccBal *acc_bal){}

void MarketPriceRetriever::OnApiTradeReport(long rec_no, const SPApiTrade *trade){}


void MarketPriceRetriever::OnPswChangeReply(long ret_code, char *ret_msg){}

void MarketPriceRetriever::OnBusinessDateReply(long business_date){}

void MarketPriceRetriever::OnApiAccountControlReply(long ret_code, char *ret_msg){}

void MarketPriceRetriever::OnApiLoadTradeReadyPush(long rec_no, const SPApiTrade *trade){}


void MarketPriceRetriever::OnApiMMOrderBeforeSendReport(SPApiMMOrder *mm_order){}

void MarketPriceRetriever::OnApiMMOrderRequestFailed(SPApiMMOrder *mm_order, long err_code, char *err_msg){}
