// (c) 2017-2018 Pttn (https://github.com/Pttn/rieMiner)

#ifndef HEADER_MINER_H
#define HEADER_MINER_H

#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <chrono>
#include "main.h"
#include "client.h"
#include "tools.h"
#include "tsqueue.hpp"

union xmmreg_t {
	uint32_t v[4];
	uint64_t v64[2];
	__m128i m128;
};

#define PENDING_SIZE 16

#define WORK_DATAS 2
#define WORK_INDEXES 64
#define GPU_WORK_INDEXES 1024
enum JobType {TYPE_CHECK, TYPE_MOD, TYPE_SIEVE};

struct MinerParameters {
	uint64_t primorialNumber;
	int16_t threads;
	uint8_t tuples;
	uint64_t sieve;
	bool solo;
	int sieveWorkers;
	uint64_t sieveBits, sieveSize, sieveWords, maxIncrements, maxIter, primorialOffset, denseLimit;
	std::vector<uint64_t> primes, inverts, modPrecompute, primeTupleOffset;

	MinerParameters() {
		primorialNumber = 40;
		threads         = 8;
		tuples          = 6;
		sieve           = 2147483648;
		sieveWorkers    = 2;
		solo            = true;
		sieveBits       = 24;
		sieveSize       = 1UL << sieveBits;
		sieveWords      = sieveSize/64;
		maxIncrements   = (1ULL << 29),
		maxIter         = maxIncrements/sieveSize;
		primorialOffset = 16057;
		denseLimit      = 16384;
		primeTupleOffset = {0, 4, 2, 4, 2, 4};
	}
};

struct primeTestWork {
	JobType type;
	uint32_t workDataIndex;
	union {
		struct {
			uint64_t loop;
			uint32_t n_indexes;
			uint32_t indexes[WORK_INDEXES];
		} testWork;
		struct {
			uint64_t start;
			uint64_t end;
		} modWork;
		struct {
			uint64_t start;
			uint64_t end;
			uint32_t sieveId;
		} sieveWork;
	};
};

struct gpuTestWork {
	uint32_t workDataIndex;
	struct {
		uint64_t loop;
		uint32_t n_indexes;
		uint32_t indexes[GPU_WORK_INDEXES];
	} testWork;
};

struct MinerWorkData {
	mpz_t z_verifyTarget, z_verifyRemainderPrimorial;
	WorkData verifyBlock;
	uint64_t outstandingTests = 0;
};

class Miner {
	std::shared_ptr<WorkManager> _manager;
	bool _inited;
	volatile uint32_t _currentHeight;
	MinerParameters _parameters;

	ts_queue<primeTestWork, 4096> _verifyWorkQueue;
	ts_queue<gpuTestWork, 1024> _gpuWorkQueue;
	ts_queue<uint64_t, 1024> _modDoneQueue;
	ts_queue<uint32_t, 128> _sieveDoneQueue;
	ts_queue<uint32_t, 4096> _testDoneQueue;
	mpz_t _primorial;
	uint64_t _nPrimes, _entriesPerSegment, _primeTestStoreOffsetsSize, _startingPrimeIndex, _nDense, _nSparse;
	uint8_t  **_sieves;
	uint32_t **_segmentHits;
	std::vector<uint64_t> _segmentCounts;
	std::vector<uint64_t> _halfPrimeTupleOffset;

	std::chrono::microseconds _modTime, _sieveTime, _verifyTime;

	bool _masterExists;
	bool _gpuExists;
	std::mutex _masterLock, _bucketLock;

	uint64_t _curWorkDataIndex;
	MinerWorkData _workData[WORK_DATAS];

	void _initPending(uint32_t pending[PENDING_SIZE]) {
		for (int i(0) ; i < PENDING_SIZE; i++) pending[i] = 0;
	}

	void _addToPending(uint8_t *sieve, uint32_t pending[PENDING_SIZE], uint64_t &pos, uint32_t ent) {
		__builtin_prefetch(&(sieve[ent >> 3]));
		uint32_t old = pending[pos];
		if (old != 0) {
			assert(old < _parameters.sieveSize);
			sieve[old >> 3] |= (1 << (old & 7));
		}
		pending[pos] = ent;
		pos++;
		pos &= PENDING_SIZE - 1;
	}

	void _addRegToPending(uint8_t *sieve, uint32_t pending[PENDING_SIZE], uint64_t &pos, xmmreg_t reg, int mask) {
		if (mask & 0x0008) _addToPending(sieve, pending, pos, reg.v[0]);
		if (mask & 0x0080) _addToPending(sieve, pending, pos, reg.v[1]);
		if (mask & 0x0800) _addToPending(sieve, pending, pos, reg.v[2]);
		if (mask & 0x8000) _addToPending(sieve, pending, pos, reg.v[3]);
	}

	void _termPending(uint8_t *sieve, uint32_t pending[PENDING_SIZE]) {
		for (uint64_t i(0) ; i < PENDING_SIZE ; i++) {
			uint32_t old(pending[i]);
			if (old != 0) {
				assert(old < _parameters.sieveSize);
				sieve[old >> 3] |= (1 << (old & 7));
			}
		}
	}

	void _putOffsetsInSegments(uint64_t *offsets, int n_offsets);
	void _updateRemainders(uint32_t workDataIndex, uint64_t start_i, uint64_t end_i);
	void _processSieve(uint8_t *sieve, uint64_t start_i, uint64_t end_i);
	void _processSieve6(uint8_t *sieve, uint64_t start_i, uint64_t end_i);
	void _verifyThread();
	bool _testPrimesGpu(struct PrimeTestCxt* gpuContext, uint32_t indexes[GPU_WORK_INDEXES], uint32_t isPrime[GPU_WORK_INDEXES], uint32_t listSize, mpz_t z_ploop, mpz_t z_temp, struct GpuTestContext* testContext);
	void _gpuThread();
	void _getTargetFromBlock(mpz_t z_target, const WorkData& block);
	void _processOneBlock(uint32_t workDataIndex, uint8_t* sieve);

	public:
	Miner(const std::shared_ptr<WorkManager> &manager) {
		_manager = manager;
		_inited = false;
		_currentHeight = 0;
		_parameters = MinerParameters();
		_sieves = NULL;
		_segmentHits = NULL;
		_nPrimes = 0;
		_entriesPerSegment = 0;
		_segmentCounts = std::vector<uint64_t>();
		_primeTestStoreOffsetsSize = 0;
		_startingPrimeIndex = 0;
		_nDense  = 0;
		_nSparse = 0;
		_masterExists = false;
		_gpuExists = false;
	}

	void init();
	void process(WorkData block);
	bool inited() {return _inited;}
	void updateHeight(uint32_t height) {_currentHeight = height;}

	void finishGpuTests(struct GpuTestContext* cxt);
};

#endif
