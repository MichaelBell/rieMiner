// (c) 2017-2020 Pttn and contributors (https://github.com/Pttn/rieMiner)

#ifndef HEADER_Miner_hpp
#define HEADER_Miner_hpp

#include <atomic>
#include <cassert>
#include "tsQueue.hpp"
#include "WorkManager.hpp"

class WorkManager;
struct WorkData;

union xmmreg_t {
	uint32_t v[4];
	uint64_t v64[2];
	__m128i m128;
};

#define PENDING_SIZE 16

#define WORK_DATAS 2
#define WORK_INDEXES 64
enum JobType {TYPE_CHECK, TYPE_MOD, TYPE_SIEVE, TYPE_DUMMY};

struct MinerParameters {
	int16_t threads;
	uint8_t tupleLengthMin;
	uint64_t primorialNumber, primeTableLimit;
	bool solo;
	bool saveRemainders;
	int sieveWorkers;
	uint64_t sieveBits, sieveSize, sieveWords, maxIncrements, maxIter;
	std::vector<uint32_t> primes, inverts, remainders;
	std::vector<uint64_t> primesHi, invertsHi, remaindersHi, modPrecompute, primeTupleOffset, primorialOffsets;
	
	MinerParameters() :
		threads(8),
		tupleLengthMin(6),
		primorialNumber(40), primeTableLimit(2147483648),
		solo(true),
		saveRemainders(false),
		sieveWorkers(2),
		sieveBits(25), sieveSize(1UL << sieveBits), sieveWords(sieveSize/64), maxIncrements(1ULL << 29), maxIter(maxIncrements/sieveSize),
		primeTupleOffset{0, 4, 2, 4, 2, 4},
		primorialOffsets{4209995887ull, 4209999247ull, 4210002607ull, 4210005967ull,
		                 7452755407ull, 7452758767ull, 7452762127ull, 7452765487ull,
		                 8145217177ull, 8145220537ull, 8145223897ull, 8145227257ull} {}

};

struct primeTestWork {
	JobType type;
	uint32_t workDataIndex;
	union {
		struct {
			uint64_t loop;
			uint32_t offsetId;
			uint32_t n_indexes;
			uint32_t indexes[WORK_INDEXES];
		} testWork;
		struct {
			uint64_t start;
			uint64_t end;
			uint32_t remainderIdx;
		} modWork;
		struct {
			uint32_t sieveId;
			uint32_t offsetId;
		} sieveWork;
	};
};

struct MinerWorkData {
	mpz_t z_verifyTarget, z_verifyRemainderPrimorial;
	WorkData verifyBlock;
	std::atomic<uint64_t> outstandingTests{0};
};

struct SieveInstance {
	uint32_t id;
	std::mutex modLock;
	uint8_t *sieve = NULL;
	uint32_t **segmentHits = NULL;
	std::atomic<uint64_t> *segmentCounts = NULL;
	uint32_t *offsets = NULL;
};

class Miner {
	std::shared_ptr<WorkManager> _manager;
	bool _inited, _running;
	volatile uint32_t _currentHeight;
	MinerParameters _parameters;
	CpuID _cpuInfo;
	
	tsQueue<primeTestWork, 1024> _modWorkQueue;
	tsQueue<primeTestWork, 4096> _verifyWorkQueue;
	tsQueue<int64_t, 9216> _workDoneQueue;
	mpz_t _primorial;
	uint64_t _nPrimes, _nLoPrimes, _entriesPerSegment, _primeTestStoreOffsetsSize, _startingPrimeIndex, _sparseLimit;
	std::vector<uint64_t> _halfPrimeTupleOffset, _primorialOffsetDiff, _primorialOffsetDiffToFirst;
	SieveInstance* _sieves;

	std::chrono::microseconds _modTime, _sieveTime, _verifyTime;
	
	bool _masterExists;
	std::mutex _masterLock, _tupleFileLock;

	uint64_t _curWorkDataIndex;
	MinerWorkData _workData[WORK_DATAS];
	uint32_t _maxWorkOut;

	void _initPending(uint32_t pending[PENDING_SIZE]) {
		for (int i(0) ; i < PENDING_SIZE; i++) pending[i] = 0;
	}

	void _addToPending(uint8_t *sieve, uint32_t pending[PENDING_SIZE], uint64_t &pos, uint32_t ent) {
		__builtin_prefetch(&(sieve[ent >> 3]));
		uint32_t old(pending[pos]);
		// assert(old < _parameters.sieveSize);
#if 0
			if (old >= _parameters.sieveSize) {
				std::cerr << "_addToPending: old = " << old << " is bigger than _parameters.sieveSize = " << _parameters.sieveSize << ", which should never happen!" << std::endl;
				std::cout << "This may happen in an unstable or faulty computer. Please check your hardware or CPU/RAM frequency/voltage settings." << std::endl;
				std::cout << "If you just worked on the code, you likely broke something." << std::endl;
				std::cout << "Temporarily changing old to dummy value of " << _parameters.sieveSize - 1 << " to allow mining to continue." << std::endl;
				old = _parameters.sieveSize - 1;
			}
#endif
		sieve[old >> 3] |= (1 << (old & 7));
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
			const uint32_t old(pending[i]);
			if (old != 0) {
				assert(old < _parameters.sieveSize);
				sieve[old >> 3] |= (1 << (old & 7));
			}
		}
	}
	
	void _putOffsetsInSegments(SieveInstance& sieve, uint64_t *offsets, uint64_t* counts, int n_offsets);
	void _updateRemainders(uint32_t workDataIndex, uint64_t start_i, uint64_t end_i, uint32_t remainderIdx);
	void _processSieve(uint8_t *sieve, uint32_t* offsets, uint64_t start_i, uint64_t end_i);
	void _processSieve6(uint8_t *sieve, uint32_t* offsets, uint64_t start_i, uint64_t end_i);
	void _processSieve9(uint8_t *sieve, uint32_t* offsets, uint64_t start_i, uint64_t end_i);
	void _runSieve(SieveInstance& sieve, uint32_t workDataIndex, uint32_t offsetId);
	bool _testPrimesIspc(uint32_t indexes[WORK_INDEXES], uint32_t is_prime[WORK_INDEXES], mpz_t z_ploop, mpz_t z_temp, uint32_t height);
	void _verifyThread();
	void _getTargetFromBlock(mpz_t z_target, const WorkData& block);
	void _processOneBlock(uint32_t& workDataIndex, bool isNewHeight);

	uint64_t _getPrime(uint64_t i) const { 
		if (i < _nLoPrimes) return _parameters.primes[i];
		else return _parameters.primesHi[i - _nLoPrimes]; 
	}
	
	public:
	Miner(const std::shared_ptr<WorkManager> &manager) {
		_manager = manager;
		_inited  = false;
		_running = false;
		_currentHeight = 0;
		_parameters = MinerParameters();
		_nPrimes = 0;
		_entriesPerSegment = 0;
		_primeTestStoreOffsetsSize = 0;
		_startingPrimeIndex = 0;
		_sparseLimit = 0;
		_masterExists = false;
	}
	
	void init();
	void process(WorkData block);
	bool inited() {return _inited;}
	void pause() {
		_running = false;
		_currentHeight = 0;
	}
	void start() {
		_running = true;
	}
	bool running() {return _running;}
	void updateHeight(uint32_t height) {_currentHeight = height;}
};

#endif
