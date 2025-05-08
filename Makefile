CPP := g++
CPPFLAGS := -std=c++20 -Wall 
INC=-I/usr/include/eigen3


BIN_DIR = Bin/

BIN_OBJS = hw5

.PHONY: all 

all: $(BIN_OBJS)

# Generic rule to build targets
%: %.cpp
	$(CPP) $(CPPFLAGS) $(INC) -o $@ $<
	mv $@ $(BIN_DIR)

clean:
	(cd Bin; rm -f $(BIN_OBJS))
