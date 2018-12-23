CC = g++ 
CFLAGS = -c -O3 -std=c++0x -fexceptions -frtti
LD = $(CC)
LDFLAGS = 
CPPFLAGS = -D__LINUX__=1 -DBOOST_MODE=1 -DBOOST_UTILS=1 -D_DEBUG=1

CWD :=  $(shell readlink -en $(dir $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))))

LIB_PATH := -L$(CWD)/lib
LIB := -lapiwrapper

INC := -I./
INC += -I$(CWD)/include

MODULE = main 

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

CPP_FILES = main.cpp MarketPriceRetriever.cpp Chameleon.cpp ConfigFile.cpp
OBJECTS = $(patsubst %.cpp, %.o, $(CPP_FILES))

all: $(MODULE)

$(MODULE): $(OBJECTS)
	$(LD) $(INC) $(LDFLAGS) $(LIB_PATH) -o $@ $^ $(LIB)
# -lrt -lboost_system -lboost_filesystem -lboost_chrono For chrono time
    
%.o: %.cpp
	$(CC) $(INC) $(CFLAGS) $(CPPFLAGS) $^ -o $@

clean:
	rm -rf $(CWD)/product
	rm -rf $(CWD)/Log
	rm $(TARGET) $(OBJECTS) $(MODULE)
