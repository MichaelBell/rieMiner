CXX    = g++
M4     = m4
AS     = as
SED    = sed
CFLAGS = -Wall -Wextra -std=gnu++11 -O3 -march=native -fno-pie -no-pie

msys_version := $(if $(findstring Msys, $(shell uname -o)),$(word 1, $(subst ., ,$(shell uname -r))),0)
ifneq ($(msys_version), 0)
LIBS   = -pthread -ljansson -lcurl -lcrypto -lgmpxx -lgmp -lws2_32 -lOpenCL -Wl,--image-base -Wl,0x10000000
MOD_1_4_ASM = mod_1_4_win.asm
else
LIBS   = -pthread -ljansson -lcurl -lcrypto -Wl,-Bstatic -lgmpxx -lgmp -Wl,-Bdynamic
MOD_1_4_ASM = mod_1_4.asm
endif

all: rieMiner

debug: CFLAGS += -g
debug: rieMiner

static: CFLAGS += -D CURL_STATICLIB -I incs/
static: LIBS   := -static -L libs/ $(LIBS)
static: rieMiner

rieMiner: main.o Miner.o StratumClient.o GBTClient.o Client.o WorkManager.cpp Stats.cpp tools.o mod_1_4.o mod_1_2_avx.o mod_1_2_avx2.o fermat.o primetest.o primetest512.o clprimetest.o
	$(CXX) $(CFLAGS) -o rieMiner $^ $(LIBS)

main.o: main.cpp main.hpp Miner.hpp StratumClient.hpp GBTClient.hpp Client.hpp WorkManager.hpp Stats.hpp tools.hpp tsQueue.hpp
	$(CXX) $(CFLAGS) -c -o main.o main.cpp

Miner.o: Miner.cpp Miner.hpp tsQueue.hpp
	$(CXX) $(CFLAGS) -c -o Miner.o Miner.cpp

StratumClient.o: StratumClient.cpp
	$(CXX) $(CFLAGS) -c -o StratumClient.o StratumClient.cpp

GBTClient.o: GBTClient.cpp
	$(CXX) $(CFLAGS) -c -o GBTClient.o GBTClient.cpp

Client.o: Client.cpp
	$(CXX) $(CFLAGS) -c -o Client.o Client.cpp

Stats.o: Stats.cpp
	$(CXX) $(CFLAGS) -c -o Stats.o Stats.cpp

WorkManager.o: WorkManager.cpp
	$(CXX) $(CFLAGS) -c -o WorkManager.o WorkManager.cpp

tools.o: tools.cpp
	$(CXX) $(CFLAGS) -c -o tools.o tools.cpp

fermat.o: ispc/fermat.cpp
	$(CXX) $(CFLAGS) -c -o fermat.o ispc/fermat.cpp -Wno-unused-function -Wno-unused-parameter -Wno-strict-overflow

mod_1_4.o: external/$(MOD_1_4_ASM)
	$(M4) external/$(MOD_1_4_ASM) >mod_1_4.s
	$(AS) mod_1_4.s -o mod_1_4.o
	rm mod_1_4.s

ifneq ($(msys_version), 0)
mod_1_2_avx.o: external/mod_1_2_avx.asm external/mod_1_2_avx_win.sed
	$(SED) -f external/mod_1_2_avx_win.sed <external/mod_1_2_avx.asm >mod_1_2_avx.asm
	$(M4) mod_1_2_avx.asm >mod_1_2_avx.s
	$(AS) mod_1_2_avx.s -o mod_1_2_avx.o
	rm mod_1_2_avx.s mod_1_2_avx.asm

mod_1_2_avx2.o: external/mod_1_2_avx2.asm external/mod_1_2_avx_win.sed
	$(SED) -f external/mod_1_2_avx_win.sed <external/mod_1_2_avx2.asm >mod_1_2_avx2.asm
	$(M4) mod_1_2_avx2.asm >mod_1_2_avx2.s
	$(AS) mod_1_2_avx2.s -o mod_1_2_avx2.o
	rm mod_1_2_avx2.s mod_1_2_avx2.asm
else
mod_1_2_avx.o: external/mod_1_2_avx.asm
	$(M4) external/mod_1_2_avx.asm >mod_1_2_avx.s
	$(AS) mod_1_2_avx.s -o mod_1_2_avx.o
	rm mod_1_2_avx.s

mod_1_2_avx2.o: external/mod_1_2_avx2.asm
	$(M4) external/mod_1_2_avx2.asm >mod_1_2_avx2.s
	$(AS) mod_1_2_avx2.s -o mod_1_2_avx2.o
	rm mod_1_2_avx2.s
endif

ifneq ($(msys_version), 0)
primetest.o: ispc/primetest.s ispc/primetest_win.sed
	$(SED) -f ispc/primetest_win.sed <ispc/primetest.s >primetest_win.s
	$(AS) primetest_win.s -o primetest.o
	rm primetest_win.s

primetest512.o: ispc/primetest512.s ispc/primetest_win.sed
	$(SED) -f ispc/primetest_win.sed <ispc/primetest512.s >primetest512_win.s
	$(AS) primetest512_win.s -o primetest512.o
	rm primetest512_win.s
else
primetest.o: ispc/primetest.s
	$(AS) ispc/primetest.s -o primetest.o

primetest512.o: ispc/primetest512.s
	$(AS) ispc/primetest512.s -o primetest512.o
endif

clprimetest.o: opencl/primetest.c
	$(CXX) $(CFLAGS) -c -o clprimetest.o opencl/primetest.c -Wno-unused-function -Wno-unused-parameter -Wno-strict-overflow

clean:
	rm -rf rieMiner *.o
