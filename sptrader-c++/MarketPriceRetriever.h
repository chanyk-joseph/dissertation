#include <string.h>
#include "ApiProxyWrapper.h"
#include "ApiProxyWrapperReply.h"
#include <fstream>
#include <map>

class MarketPriceRetriever :  public ApiProxyWrapperReply
{
    public:
        MarketPriceRetriever(string apiEndPoint, int apiPort, string apiLicense, string apiID, string userID, string password);
        MarketPriceRetriever(string configFilePath);
        ~MarketPriceRetriever(void);

        bool Init(string outputCSV_Path);
        void Unload();
        void InitOutputCSVForProduct(string prodCode);
        bool SubscribeProductPrice(string prodCode);
        bool UnsubsribteProductPrice(string prodCode);

        virtual void OnLoginReply(char *user_id, long ret_code,char *ret_msg);
        virtual void OnConnectedReply(long host_type, long con_status);
        virtual void OnApiOrderRequestFailed(tinyint action, const SPApiOrder *order, long err_code, char *err_msg);
        virtual void OnApiOrderReport(long rec_no, const SPApiOrder *order);
        virtual void OnApiOrderBeforeSendReport(const SPApiOrder *order);
        virtual void OnAccountLoginReply(char *accNo, long ret_code, char* ret_msg);
        virtual void OnAccountLogoutReply(char *accNo, long ret_code, char* ret_msg);
        virtual void OnAccountInfoPush(const SPApiAccInfo *acc_info);
        virtual void OnAccountPositionPush(const SPApiPos *pos);
        virtual void OnUpdatedAccountPositionPush(const SPApiPos *pos);
        virtual void OnUpdatedAccountBalancePush(const SPApiAccBal *acc_bal);
        virtual void OnApiTradeReport(long rec_no, const SPApiTrade *trade);
        virtual void OnApiPriceUpdate(const SPApiPrice *price);
        virtual void OnApiTickerUpdate(const SPApiTicker *ticker);
        virtual void OnPswChangeReply(long ret_code, char *ret_msg);
        virtual void OnProductListByCodeReply(long req_id, char *inst_code, bool is_ready, char *ret_msg);
        virtual void OnInstrumentListReply(long req_id, bool is_ready, char *ret_msg);
        virtual void OnBusinessDateReply(long business_date);
        virtual void OnApiMMOrderBeforeSendReport(SPApiMMOrder *mm_order);
        virtual void OnApiMMOrderRequestFailed(SPApiMMOrder *mm_order, long err_code, char *err_msg);
        virtual void OnApiQuoteRequestReceived(char *product_code, char buy_sell, long qty);
        virtual void OnApiAccountControlReply(long ret_code, char *ret_msg);
        virtual void OnApiLoadTradeReadyPush(long rec_no, const SPApiTrade *trade);
    private:
        string apiEndPoint;
        int apiPort;
        string apiLicense;
        string apiID;
        string userID;
        string password;

        ApiProxyWrapper apiProxyWrapper;
        bool serviceConnectionStatus[3] = {0};
        bool isAllServiceConnected();

        bool isInstrumentsListDownloaded = false;
        bool retrieveAllInstruments();

        int maxConcurrencyOfProductRetrieve = 30;
        bool retrievingFlag = false;
        bool retrieveAllProducts();

        vector<SPApiInstrument> instrumentsList;
        vector<SPApiProduct> productsList;
        vector<string> subscribedProductsList;

        string outputMarketDataFileNamePrefix;
        map<string, ofstream*> prodCodePriceOutputFileStreams;
        map<string, ofstream*> prodCodeTickerOutputFileStreams;
        string getOutputPath(string prodCode, bool isTicker);
};