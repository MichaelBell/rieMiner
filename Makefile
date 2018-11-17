CXX    = g++
M4     = m4
AS     = as
CFLAGS = -Wall -Wextra -std=gnu++11 -O3 -march=native

msys_version := $(if $(findstring Msys, $(shell uname -o)),$(word 1, $(subst ., ,$(shell uname -r))),0)
ifneq ($(msys_version), 0)
LIBS   = -pthread -ljansson -lcurl -lcrypto -lgmp -lgmpxx -lws2_32 -lOpenCL
MOD_1_4_ASM = mod_1_4_win.asm
else
LIBS   = -pthread -ljansson -lcurl -lcrypto -Wl,-Bstatic -lgmp -lgmpxx -Wl,-Bdynamic -lOpenCL
MOD_1_4_ASM = mod_1_4.asm
endif

all: rieMiner

release: CFLAGS += -DNDEBUG
release: rieMiner

debug: CFLAGS += -g
debug: rieMiner

rieMiner: main.o miner.o stratumclient.o gbtclient.o client.o tools.o mod_1_4.o primetest.o
	$(CXX) $(CFLAGS) -o rieMiner $^ $(LIBS)

main.o: main.cpp main.h miner.h client.h gbtclient.h stratumclient.h tools.h
	$(CXX) $(CFLAGS) -c -o main.o main.cpp

miner.o: miner.cpp miner.h
	$(CXX) $(CFLAGS) -c -o miner.o miner.cpp

stratumclient.o: stratumclient.cpp
	$(CXX) $(CFLAGS) -c -o stratumclient.o stratumclient.cpp

gbtclient.o: gbtclient.cpp
	$(CXX) $(CFLAGS) -c -o gbtclient.o gbtclient.cpp

client.o: client.cpp
	$(CXX) $(CFLAGS) -c -o client.o client.cpp

tools.o: tools.cpp
	$(CXX) $(CFLAGS) -c -o tools.o tools.cpp

mod_1_4.o: external/$(MOD_1_4_ASM)
	$(M4) external/$(MOD_1_4_ASM) >mod_1_4.s
	$(AS) mod_1_4.s -o mod_1_4.o
	rm mod_1_4.s

primetest.o: opencl/primetest.c
	$(CXX) $(CFLAGS) -c -o primetest.o opencl/primetest.c -Wno-unused-function -Wno-unused-parameter -Wno-strict-overflow

clean:
	rm -rf rieMiner *.o
