#include <iostream>
#include <unistd.h>
#include "MarketPriceRetriever.h"

using namespace std;

int main( int argc, const char* argv[] ){
    MarketPriceRetriever marketPriceRetriever("config.txt");


    while(1){
        getchar();
    }
    return 0;
}