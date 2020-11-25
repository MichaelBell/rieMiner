/* (c) 2017-2020 Pttn (https://github.com/Pttn/rieMiner)
(c) 2018 Michael Bell/Rockhawk (improvements of work management between threads and some more) (https://github.com/MichaelBell/) */

#include <gmpxx.h> // With Uint64_Ts, we still need to use the Mpz_ functions, otherwise there are "ambiguous overload" errors on Windows...
#include "Miner.hpp"

constexpr int factorsCacheSize(16384);
constexpr uint16_t maxSieveWorkers(16); // There is a noticeable performance penalty using Vector so we are using Arrays.
thread_local std::array<uint64_t*, maxSieveWorkers> factorsCache{nullptr};
thread_local std::array<uint64_t*, maxSieveWorkers> factorsCacheCounts{nullptr};
thread_local uint16_t threadId(65535);

void Miner::init(const MinerParameters &minerParameters) {
	_shouldRestart = false;
	if (_inited) {
		ERRORMSG("The miner is already inited");
		return;
	}
	Job job;
	if (_client == nullptr) {
		ERRORMSG("The miner cannot be initialized without a client");
		return;
	}
	else if (!_client->getJob(job, true)) {
		std::cout << "Could not get data from Client :|" << std::endl;
		return;
	}
	_difficultyAtInit = job.difficulty;
	
	std::cout << "Initializing miner..." << std::endl;
	// Get settings from Configuration File.
	_parameters = minerParameters;
	if (_parameters.threads == 0) {
		_parameters.threads = std::thread::hardware_concurrency();
		if (_parameters.threads == 0) {
			std::cout << "Could not detect number of Threads, setting to 1." << std::endl;
			_parameters.threads = 1;
		}
	}
	std::cout << "Threads: " << _parameters.threads;
	if (_parameters.primorialOffsets.size() == 0) { // Set the default Primorial Offsets if not chosen (must be chosen if the chosen pattern is not hardcoded)
		auto defaultPrimorialOffsetsIterator(std::find_if(defaultConstellationData.begin(), defaultConstellationData.end(), [this](const auto& constellationData) {return constellationData.first == _parameters.pattern;}));
		if (defaultPrimorialOffsetsIterator == defaultConstellationData.end()) {
			std::cout << std::endl << "Not hardcoded Constellation Offsets chosen and no Primorial Offset set." << std::endl;
			return;
		}
		else
			_parameters.primorialOffsets = defaultPrimorialOffsetsIterator->second;
	}
	_primorialOffsets = v64ToVMpz(_parameters.primorialOffsets);
	if (_parameters.sieveWorkers == 0) {
		double proportion;
		if (_parameters.pattern.size() >= 7) proportion = 0.85 - _difficultyAtInit/1920.;
		else if (_parameters.pattern.size() == 6) proportion = 0.75 - _difficultyAtInit/1792.;
		else if (_parameters.pattern.size() == 5) proportion = 0.7 - _difficultyAtInit/1280.;
		else if (_parameters.pattern.size() == 4) proportion = 0.5 - _difficultyAtInit/1280.;
		else proportion = 0.;
		if (proportion < 0.) proportion = 0.;
		if (job.powVersion == -1) proportion *= 2.5;
		if (proportion > 1.) proportion = 1.;
		_parameters.sieveWorkers = std::ceil(proportion*static_cast<double>(_parameters.threads));
	}
	_parameters.sieveWorkers = std::min(static_cast<int>(_parameters.sieveWorkers), static_cast<int>(_parameters.threads) - 1);
	_parameters.sieveWorkers = std::max(static_cast<int>(_parameters.sieveWorkers), 1);
	_parameters.sieveWorkers = std::min(_parameters.sieveWorkers, maxSieveWorkers);
	_parameters.sieveWorkers = std::min(static_cast<int>(_parameters.sieveWorkers), static_cast<int>(_primorialOffsets.size()));
	std::cout << " (" << _parameters.sieveWorkers << " Sieve Worker(s))" << std::endl;
	
	std::vector<uint64_t> cumulativeOffsets(_parameters.pattern.size(), 0);
	std::partial_sum(_parameters.pattern.begin(), _parameters.pattern.end(), cumulativeOffsets.begin(), std::plus<uint64_t>());
	std::cout << "Constellation pattern: n + (" << formatContainer(cumulativeOffsets) << "), length " << _parameters.pattern.size() << std::endl;
	if (_mode == "Search") {
		if (_parameters.tupleLengthMin < 1 || _parameters.tupleLengthMin > _parameters.pattern.size())
			_parameters.tupleLengthMin = std::max(1, static_cast<int>(_parameters.pattern.size()) - 1);
		std::cout << "Will show tuples of at least length " << _parameters.tupleLengthMin << std::endl;
	}
	
	if (_parameters.primeTableLimit == 0) {
		constexpr uint32_t primeTableLimitMax(1073741824U);
		_parameters.primeTableLimit = std::pow(_difficultyAtInit, 6.)/std::pow(2., 3.*static_cast<double>(_parameters.pattern.size()) + 7.);
		if (_parameters.threads > 16) {
			_parameters.primeTableLimit *= 16;
			_parameters.primeTableLimit /= static_cast<double>(_parameters.threads);
		}
		_parameters.primeTableLimit = std::min(_parameters.primeTableLimit, primeTableLimitMax);
	}
	std::cout << "Prime Table Limit: " << _parameters.primeTableLimit << std::endl;
	std::transform(_parameters.pattern.begin(), _parameters.pattern.end(), std::back_inserter(_halfPattern), [](uint64_t n) {return n >> 1;});
	
	uint32_t primeTableFileBytes, savedPrimes(0), largestSavedPrime;
	std::fstream file(primeTableFile);
	if (file) {
		file.seekg(0, std::ios::end);
		primeTableFileBytes = file.tellg();
		savedPrimes = primeTableFileBytes/sizeof(decltype(_primes)::value_type);
		if (savedPrimes > 0) {
			file.seekg(-static_cast<int64_t>(sizeof(decltype(_primes)::value_type)), std::ios::end);
			file.read(reinterpret_cast<char*>(&largestSavedPrime), sizeof(decltype(_primes)::value_type));
		}
	}
	std::chrono::time_point<std::chrono::steady_clock> t0(std::chrono::steady_clock::now());
	if (savedPrimes > 0 && _parameters.primeTableLimit >= 1048576 && _parameters.primeTableLimit <= largestSavedPrime) {
		std::cout << "Extracting prime numbers from " << primeTableFile << " (" << primeTableFileBytes << " bytes, " << savedPrimes << " primes, largest " << largestSavedPrime << ")..." << std::endl;
		uint64_t nPrimesUpperBound(std::min(1.085*static_cast<double>(_parameters.primeTableLimit)/std::log(static_cast<double>(_parameters.primeTableLimit)), static_cast<double>(savedPrimes))); // 1.085 = max(π(p)log(p)/p) for p >= 2^20
		try {
			_primes = std::vector<uint32_t>(nPrimesUpperBound);
		}
		catch (std::bad_alloc& ba) {
			ERRORMSG("Unable to allocate memory for the prime table");
			_suggestLessMemoryIntensiveOptions(_parameters.primeTableLimit/8, _parameters.sieveWorkers);
			return;
		}
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(_primes.data()), nPrimesUpperBound*sizeof(decltype(_primes)::value_type));
		file.close();
		for (auto i(_primes.size() - 1) ; i > 0 ; i--) {
			if (_primes[i] <= _parameters.primeTableLimit) {
				_primes.resize(i + 1);
				break;
			}
		}
		std::cout << _primes.size() << " first primes extracted in " << timeSince(t0) << " s (" << _primes.size()*sizeof(decltype(_primes)::value_type) << " bytes)." << std::endl;
	}
	else {
		std::cout << "Generating prime table using sieve of Eratosthenes..." << std::endl;
		try {
			_primes = generatePrimeTable(_parameters.primeTableLimit);
		}
		catch (std::bad_alloc& ba) {
			ERRORMSG("Unable to allocate memory for the prime table");
			_suggestLessMemoryIntensiveOptions(_parameters.primeTableLimit/8, _parameters.sieveWorkers);
			return;
		}
		std::cout << "Table with all " << _primes.size() << " first primes generated in " << timeSince(t0) << " s (" << _primes.size()*sizeof(decltype(_primes)::value_type) << " bytes)." << std::endl;
	}
	_nPrimes = _primes.size();
	
	if (_parameters.sieveBits == 0)
		_parameters.sieveBits = _parameters.sieveWorkers <= 4 ? 23 : 22;
	_parameters.sieveSize = 1 << _parameters.sieveBits;
	_parameters.sieveWords = _parameters.sieveSize/64;
	std::cout << "Sieve Size: " << "2^" << _parameters.sieveBits << " = " << _parameters.sieveSize << " (" << _parameters.sieveWords << " words)" << std::endl;
	if (_parameters.sieveIterations == 0)
		_parameters.sieveIterations = 16;
	std::cout << "Sieve Iterations: " << _parameters.sieveIterations << std::endl;
	_factorMax = _parameters.sieveIterations*_parameters.sieveSize;
	std::cout << "Primorial Factor Max: " << _factorMax << std::endl;
	
	uint32_t bitsForOffset;
	// The primorial times the maximum factor should be smaller than the allowed limit for the target offset.
	if (_mode == "Solo" || _mode == "Pool" || _mode == "Test") {
		bitsForOffset = std::floor(_difficultyAtInit - 265.); // 1 . leading 8 bits . hash (256 bits) . remaining bits for the offset
		bitsForOffset -= 48; // Some margin to take in account the Difficulty fluctuations
	}
	else if (_mode == "Search")
		bitsForOffset = std::floor(_difficultyAtInit - 97.); // 1 . leading 16 bits . random 80 bits . remaining bits for the offset
	else
		bitsForOffset = std::floor(_difficultyAtInit - 81.); // 1 . leading 16 bits . constructed 64 bits . remaining bits for the offset
	if (job.powVersion == -1) // Maximum 256 bits allowed before the fork
		bitsForOffset = std::min(bitsForOffset, 256U);
	mpz_class primorialLimit(1);
	primorialLimit <<= bitsForOffset;
	primorialLimit /= u64ToMpz(_factorMax);
	if (primorialLimit == 0) {
		std::cout << "The Difficulty is too low. Try to increase it or decrease the Sieve Size/Iterations." << std::endl;
		std::cout << "Available digits for the offsets: " << bitsForOffset << std::endl;
		return;
	}
	mpz_set_ui(_primorial.get_mpz_t(), 1);
	for (uint64_t i(0) ; i < _primes.size() ; i++) {
		if (i == _parameters.primorialNumber && _parameters.primorialNumber != 0)
			break;
		else {
			if (_primorial*static_cast<uint32_t>(_primes[i]) >= primorialLimit) {
				if (_parameters.primorialNumber != 0)
					std::cout << "The provided Primorial Number " <<_parameters.primorialNumber  << " is too large and will be reduced." << std::endl;
				_parameters.primorialNumber = i;
				break;
			}
		}
		_primorial *= static_cast<uint32_t>(_primes[i]);
		if (i + 1 == _primes.size())
			_parameters.primorialNumber = i + 1;
	}
	std::cout << "Primorial Number: " << _parameters.primorialNumber << std::endl;
	std::cout << "Primorial: p" << _parameters.primorialNumber << "# = " << _primes[_parameters.primorialNumber - 1] << "# = ";
	if (mpz_sizeinbase(_primorial.get_mpz_t(), 10) < 18)
		std::cout << _primorial;
	else
		std::cout << "~" << _primorial.get_str()[0] << "." << _primorial.get_str().substr(1, 12) << "*10^" << _primorial.get_str().size() - 1;
	std::cout << " (" << mpz_sizeinbase(_primorial.get_mpz_t(), 2) << " bits)" << std::endl;
	std::cout << "Primorial Offsets: " << formatContainer(_primorialOffsets) << std::endl;
	_primorialOffsetDiff.resize(_parameters.sieveWorkers - 1);
	const uint64_t constellationDiameter(cumulativeOffsets.back());
	for (int j(1) ; j < _parameters.sieveWorkers ; j++)
		_primorialOffsetDiff[j - 1] = _parameters.primorialOffsets[j] - _parameters.primorialOffsets[j - 1] - constellationDiameter;
	
	uint64_t additionalFactorsCountEstimation(0); // tupleSize*factorMax*(sum of 1/p, for p in the prime table >= factorMax); it is the estimation of how many such p will eliminate a factor (factorMax/p being the probability of the modulo p being < factorMax)
	double sumInversesOfPrimes(0.);
	_primesIndexThreshold = 0; // Number of prime numbers smaller than factorMax in the table
	for (uint64_t i(0) ; i < _nPrimes ; i++) {
		const uint64_t p(_primes[i]);
		if (p >= _factorMax) {
			if (_primesIndexThreshold == 0)
				_primesIndexThreshold = i;
			sumInversesOfPrimes += 1./static_cast<double>(p);
		}
	}
	if (_primesIndexThreshold == 0)
		_primesIndexThreshold = _nPrimes;
	std::cout << "Prime index threshold: " << _primesIndexThreshold << std::endl;
	const uint64_t factorsToEliminateEntries(_parameters.pattern.size()*_primesIndexThreshold); // PatternLength entries for every prime < factorMax
	additionalFactorsCountEstimation = _parameters.pattern.size()*ceil(static_cast<double>(_factorMax)*sumInversesOfPrimes);
	const uint64_t additionalFactorsEntriesPerIteration(17ULL*(additionalFactorsCountEstimation/_parameters.sieveIterations)/16ULL + 64ULL); // Have some margin
	std::cout << "Estimated additional factors: " << additionalFactorsCountEstimation << " (allocated per iteration: " << additionalFactorsEntriesPerIteration << ")" << std::endl;
	{
		std::cout << "Precomputing modular inverses..." << std::endl; // The precomputed data is used to speed up computations in _doPresieveTask.
		t0 = std::chrono::steady_clock::now();
		try {
			_modularInverses.resize(_nPrimes); // Table of inverses of the primorial modulo a prime number in the table with index >= primorialNumber.
		}
		catch (std::bad_alloc& ba) {
			ERRORMSG("Unable to allocate memory for the precomputed data");
			_suggestLessMemoryIntensiveOptions(_parameters.primeTableLimit/4, _parameters.sieveWorkers);
			return;
		}
		const uint64_t blockSize((_nPrimes - _parameters.primorialNumber + _parameters.threads - 1)/_parameters.threads);
		std::thread threads[_parameters.threads];
		for (uint16_t j(0) ; j < _parameters.threads ; j++) {
			threads[j] = std::thread([&, j]() {
				mpz_class modularInverse, prime;
				const uint64_t endIndex(std::min(_parameters.primorialNumber + (j + 1)*blockSize, _nPrimes));
				for (uint64_t i(_parameters.primorialNumber + j*blockSize) ; i < endIndex ; i++) {
					mpz_set_ui(prime.get_mpz_t(), _primes[i]);
					mpz_invert(modularInverse.get_mpz_t(), _primorial.get_mpz_t(), prime.get_mpz_t()); // modularInverse*primorial ≡ 1 (mod prime)
					_modularInverses[i] = mpz_get_ui(modularInverse.get_mpz_t());
				}
			});
		}
		for (uint16_t j(0) ; j < _parameters.threads ; j++) threads[j].join();
		std::cout << "Tables of " << _modularInverses.size() - _parameters.primorialNumber << " modular inverses generated in " << timeSince(t0) << " s (" << (_modularInverses.size() - 2*_parameters.primorialNumber)*sizeof(decltype(_modularInverses)::value_type) << " bytes)." << std::endl;
	}
	
	try {
		std::vector<Sieve> sieves(_parameters.sieveWorkers);
		_sieves.swap(sieves);
		for (std::vector<Sieve>::size_type i(0) ; i < _sieves.size() ; i++) {
			_sieves[i].id = i;
			_sieves[i].additionalFactorsToEliminateCounts = new std::atomic<uint64_t>[_parameters.sieveIterations];
		}
		std::cout << "Allocating " << sizeof(uint64_t)*_parameters.sieveWorkers*_parameters.sieveWords << " bytes for the primorial factors tables..." << std::endl;
		for (auto &sieve : _sieves)
			sieve.factorsTable = new uint64_t[_parameters.sieveWords];
	}
	catch (std::bad_alloc& ba) {
		ERRORMSG("Unable to allocate memory for the primorial factors tables");
		_suggestLessMemoryIntensiveOptions(_parameters.primeTableLimit/3, _parameters.sieveWorkers);
		return;
	}
	
	try {
		std::cout << "Allocating " << sizeof(uint32_t)*_parameters.sieveWorkers*factorsToEliminateEntries << " bytes for the primorial factors..." << std::endl;
		for (auto &sieve : _sieves) {
			sieve.factorsToEliminate = new uint32_t[factorsToEliminateEntries];
			memset(sieve.factorsToEliminate, 0, sizeof(uint32_t)*factorsToEliminateEntries);
		}
	}
	catch (std::bad_alloc& ba) {
		ERRORMSG("Unable to allocate memory for the primorial factors");
		_suggestLessMemoryIntensiveOptions(_parameters.primeTableLimit/2, std::max(static_cast<int>(_parameters.sieveWorkers) - 1, 1));
		return;
	}
	
	try {
		std::cout << "Allocating " << sizeof(uint32_t)*_parameters.sieveWorkers*_parameters.sieveIterations*additionalFactorsEntriesPerIteration << " bytes for the additional primorial factors..." << std::endl;
		for (auto &sieve : _sieves) {
			sieve.additionalFactorsToEliminate = new uint32_t*[_parameters.sieveIterations];
			for (uint64_t j(0) ; j < _parameters.sieveIterations ; j++)
				sieve.additionalFactorsToEliminate[j] = new uint32_t[additionalFactorsEntriesPerIteration];
		}
	}
	catch (std::bad_alloc& ba) {
		ERRORMSG("Unable to allocate memory for the additional primorial factors");
		_suggestLessMemoryIntensiveOptions(2*_parameters.primeTableLimit/3, std::max(static_cast<int>(_parameters.sieveWorkers) - 1, 1));
		return;
	}
	// Initial guess at a value for the threshold
	_nRemainingCheckTasksThreshold = 32U*_parameters.threads*_parameters.sieveWorkers;
	_inited = true;
	std::cout << "Done initializing miner." << std::endl;
}

void Miner::startThreads() {
	if (!_inited)
		ERRORMSG("The miner is not inited");
	else if (_client == nullptr)
		ERRORMSG("The miner cannot start mining without a client");
	else if (_running)
		ERRORMSG("The miner is already running");
	else {
		_running = true;
		_statManager.start(_parameters.pattern.size());
		std::cout << "Starting the miner's master thread..." << std::endl;
		_masterThread = std::thread(&Miner::_manageTasks, this);
		std::cout << "Starting " << _parameters.threads << " miner's worker threads..." << std::endl;
		for (uint16_t i(0) ; i < _parameters.threads ; i++)
			_workerThreads.push_back(std::thread(&Miner::_doTasks, this, i));
		std::cout << "-----------------------------------------------------------" << std::endl;
		if (_mode == "Benchmark" || _mode == "Search")
			std::cout << Stats::formattedTime(_statManager.timeSinceStart()) << " Started " << _mode << ", difficulty " << FIXED(3) << _client->currentDifficulty() << std::endl;
		else
			std::cout << Stats::formattedClockTimeNow() << " Started mining at block " << _client->currentHeight() << ", difficulty " << FIXED(3) << _client->currentDifficulty() << std::endl;
	}
}

void Miner::stopThreads() {
	if (!_running)
		ERRORMSG("The miner is already not running");
	else {
		_running = false;
		if (_mode == "Benchmark" || _mode == "Search")
			printTupleStats();
		std::cout << "Waiting for the miner's master thread to finish..." << std::endl;
		_tasksDoneInfos.push_front(TaskDoneInfo{Task::Type::Dummy, .empty = {}}); // Unblock if master thread stuck in blocking_pop_front().
		_masterThread.join();
		std::cout << "Waiting for the miner's worker threads to finish..." << std::endl;
		for (uint16_t i(0) ; i < _parameters.threads ; i++)
			_tasks.push_front(Task{Task::Type::Dummy, 0, {}}); // Unblock worker threads stuck in blocking_pop_front().
		for (auto &workerThread : _workerThreads)
			workerThread.join();
		_workerThreads.clear();
		std::cout << "Miner threads stopped." << std::endl;
		_presieveTasks.clear();
		_tasks.clear();
		_tasksDoneInfos.clear();
		for (auto &work : _works) work.clear();
	}
}

void Miner::clear() {
	if (_running)
		ERRORMSG("Cannot clear the miner while it is running");
	else if (!_inited)
		ERRORMSG("Cannot clear the miner if it is not inited");
	else {
		std::cout << "Clearing miner's data..." << std::endl;
		_inited = false;
		for (auto &sieve : _sieves) {
			delete sieve.factorsTable;
			delete sieve.factorsToEliminate;
			for (uint64_t j(0) ; j < _parameters.sieveIterations ; j++)
				delete sieve.additionalFactorsToEliminate[j];
			delete sieve.additionalFactorsToEliminate;
			delete sieve.additionalFactorsToEliminateCounts;
		}
		_sieves.clear();
		_primes.clear();
		_modularInverses.clear();
		_primorialOffsets.clear();
		_halfPattern.clear();
		_primorialOffsetDiff.clear();
		_parameters = MinerParameters();
		std::cout << "Miner's data cleared." << std::endl;
	}
}

void Miner::_addCachedAdditionalFactorsToEliminate(Sieve& sieve, uint64_t *factorsCache, uint64_t *factorsCacheCounts, const int factorsCacheTotalCount) {
	for (uint64_t i(0) ; i < _parameters.sieveIterations ; i++) // Initialize the counts for use as index and update the sieve's one
		factorsCacheCounts[i] = sieve.additionalFactorsToEliminateCounts[i].fetch_add(factorsCacheCounts[i]);
	for (int i(0) ; i < factorsCacheTotalCount ; i++) {
		const uint64_t factor(factorsCache[i]),
		               sieveIteration(factor >> _parameters.sieveBits),
		               indexInFactorsTable(factorsCacheCounts[sieveIteration]);
		sieve.additionalFactorsToEliminate[sieveIteration][indexInFactorsTable] = factor & (_parameters.sieveSize - 1); // factor % sieveSize
		factorsCacheCounts[sieveIteration]++;
	}
	for (uint64_t i(0) ; i < _parameters.sieveIterations ; i++)
		factorsCacheCounts[i] = 0;
}

void Miner::_doPresieveTask(const Task &task) {
	const uint64_t workIndex(task.workIndex), firstPrimeIndex(task.presieve.start), lastPrimeIndex(task.presieve.end);
	const mpz_class firstCandidate(_works[workIndex].primorialMultipleStart + _primorialOffsets[0]);
	std::array<int, maxSieveWorkers> factorsCacheTotalCounts{0};
	std::array<uint64_t*, maxSieveWorkers> &factorsCacheRef(factorsCache), // On Windows, caching these thread_local pointers on the stack makes a noticeable perf difference.
	                                       &factorsCacheCountsRef(factorsCacheCounts);
	const uint64_t tupleSize(_parameters.pattern.size());
	
	for (uint64_t i(firstPrimeIndex) ; i < lastPrimeIndex ; i++) {
		const uint64_t p(_primes[i]);
		uint64_t mi[4];
		mi[0] = _modularInverses[i]; // Modular inverse of the primorial: mi[0]*primorial ≡ 1 (mod p). The modularInverses were precomputed in init().
		mi[1] = (mi[0] << 1); // mi[i] = (2*i*mi[0]) % p for i > 0.
		if (mi[1] >= p) mi[1] -= p;
		mi[2] = mi[1] << 1;
		if (mi[2] >= p) mi[2] -= p;
		mi[3] = mi[1] + mi[2];
		if (mi[3] >= p) mi[3] -= p;
		// Compute the first eliminated primorial factor for p fp.
		// fp is the solution of firstCandidate + primorial*f ≡ 0 (mod p) for 0 <= f < p: fp = (p - (firstCandidate % p))*mi[0] % p.
		// In the sieving phase, numbers of the form firstCandidate + (p*i + fp)*primorial for 0 <= i < factorMax are eliminated as they are divisible by p.
		// This is for the first number of the constellation. Later, the mi[1-3] will be used to adjust fp for the other elements of the constellation.
		const uint64_t remainder(mpz_tdiv_ui(firstCandidate.get_mpz_t(), p)), pa(p - remainder);
		uint64_t fp((pa*mi[0]) % p);
		
		// We use a macro here to ensure the compiler inlines the code, and also make it easier to early
		// out of the function completely if the current height has changed.
#define addFactorsToEliminateForP(sieveWorkerIndex) {						                                                   \
			if (i < _primesIndexThreshold) {			                                                                       \
				_sieves[sieveWorkerIndex].factorsToEliminate[tupleSize*i] = fp;		                                           \
				for (std::vector<uint64_t>::size_type f(1) ; f < _halfPattern.size() ; f++) {		                           \
					if (fp < mi[_halfPattern[f]]) fp += p;	                                                                   \
					fp -= mi[_halfPattern[f]];	                                                                               \
					_sieves[sieveWorkerIndex].factorsToEliminate[tupleSize*i + f] = fp;	                                       \
				}		                                                                                                       \
			}			                                                                                                       \
			else {			                                                                                                   \
				if (factorsCacheTotalCounts[sieveWorkerIndex] + _halfPattern.size() >= factorsCacheSize) {		           \
					if (_works[workIndex].job.height != _client->currentHeight())	                                           \
						return;                                                                                                \
					_addCachedAdditionalFactorsToEliminate(_sieves[sieveWorkerIndex], factorsCacheRef[sieveWorkerIndex], factorsCacheCountsRef[sieveWorkerIndex], factorsCacheTotalCounts[sieveWorkerIndex]); \
					factorsCacheTotalCounts[sieveWorkerIndex] = 0;	                                                   \
				}		                                                                                                       \
				if (fp < _factorMax) {		                                                                                   \
					factorsCacheRef[sieveWorkerIndex][factorsCacheTotalCounts[sieveWorkerIndex]++] = fp;	   \
					factorsCacheCountsRef[sieveWorkerIndex][fp >> _parameters.sieveBits]++;	                       \
				}		                                                                                                       \
				for (std::vector<uint64_t>::size_type f(1) ; f < _halfPattern.size() ; f++) {		                           \
					if (fp < mi[_halfPattern[f]]) fp += p;	                                                                   \
					fp -= mi[_halfPattern[f]];	                                                                               \
					if (fp < _factorMax) {	                                                                                   \
						factorsCacheRef[sieveWorkerIndex][factorsCacheTotalCounts[sieveWorkerIndex]++] = fp; \
						factorsCacheCountsRef[sieveWorkerIndex][fp >> _parameters.sieveBits]++;                    \
					}	                                                                                                       \
				}		                                                                                                       \
			}		                                                                                                           \
		};
		addFactorsToEliminateForP(0);
		if (_parameters.sieveWorkers == 1) continue;
		
		// Recompute fp to adjust to the PrimorialOffsets of other Sieve Workers.
		uint64_t r((_primorialOffsetDiff[0]*mi[0]) % p);
		if (fp < r) fp += p;
		fp -= r;
		addFactorsToEliminateForP(1);
		
		for (int j(2) ; j < _parameters.sieveWorkers ; j++) {
			if (_primorialOffsetDiff[j - 1] != _primorialOffsetDiff[j - 2])
				r = (_primorialOffsetDiff[j - 1]*mi[0]) % p;
			if (fp < r) fp += p;
			fp -= r;
			addFactorsToEliminateForP(j);
		}
	}
	
	if (lastPrimeIndex > _primesIndexThreshold) {
		for (int j(0) ; j < _parameters.sieveWorkers ; j++) {
			if (factorsCacheTotalCounts[j] > 0) {
				_addCachedAdditionalFactorsToEliminate(_sieves[j], factorsCacheRef[j], factorsCacheCountsRef[j], factorsCacheTotalCounts[j]);
				factorsCacheTotalCounts[j] = 0;
			}
		}
	}
}

void Miner::_doSieveTask(Task task) {
	Sieve& sieve(_sieves[task.sieve.id]);
	std::unique_lock<std::mutex> presieveLock(sieve.presieveLock, std::defer_lock);
	const uint64_t workIndex(task.workIndex), sieveIteration(task.sieve.iteration), firstPrimeIndex(_parameters.primorialNumber);
	const uint64_t tupleSize(_parameters.pattern.size());
	std::array<uint32_t, sieveCacheSize> sieveCache{0};
	uint64_t sieveCachePos(0);
	Task checkTask{Task::Type::Check, workIndex, .check = {}};
	
	if (_works[workIndex].job.height != _client->currentHeight()) // Abort Sieve Task if new block (but count as Task done)
		goto sieveEnd;
	
	memset(sieve.factorsTable, 0, sizeof(uint64_t)*_parameters.sieveWords);
	
	// Eliminate the p*i + fp factors (p < factorMax).
	for (uint64_t i(firstPrimeIndex) ; i < _primesIndexThreshold ; i++) {
		const uint32_t p(_primes[i]);
		for (uint64_t f(0) ; f < tupleSize; f++) {
			while (sieve.factorsToEliminate[i*tupleSize + f] < _parameters.sieveSize) { // Eliminate primorial factors of the form p*m + fp for every m*p in the current table.
				_addToSieveCache(sieve.factorsTable, sieveCache, sieveCachePos, sieve.factorsToEliminate[i*tupleSize + f]);
				sieve.factorsToEliminate[i*tupleSize + f] += p; // Increment the m
			}
			sieve.factorsToEliminate[i*tupleSize + f] -= _parameters.sieveSize; // Prepare for the next iteration
		}
	}
	_endSieveCache(sieve.factorsTable, sieveCache);
	
	if (_works[workIndex].job.height != _client->currentHeight())
		goto sieveEnd;
	
	// Wait for the presieve tasks that generate the additional factors to finish.
	if (sieveIteration == 0) presieveLock.lock();
	
	// Eliminate these factors.
	sieveCache = std::array<uint32_t, sieveCacheSize>{0};
	sieveCachePos = 0;
	for (uint64_t i(0) ; i < sieve.additionalFactorsToEliminateCounts[sieveIteration] ; i++)
		_addToSieveCache(sieve.factorsTable, sieveCache, sieveCachePos, sieve.additionalFactorsToEliminate[sieveIteration][i]);
	_endSieveCache(sieve.factorsTable, sieveCache);
	
	if (_works[workIndex].job.height != _client->currentHeight())
		goto sieveEnd;
	
	checkTask.check.nCandidates = 0;
	checkTask.check.offsetId = sieve.id;
	checkTask.check.factorStart = sieveIteration*_parameters.sieveSize;
	// Extract candidates from the sieve and create verify tasks of up to maxCandidatesPerCheckTask candidates.
	for (uint32_t b(0) ; b < _parameters.sieveWords ; b++) {
		uint64_t sieveWord(~sieve.factorsTable[b]); // ~ is the Bitwise Not: ones then indicate the candidates and zeros the previously eliminated numbers.
		while (sieveWord != 0) {
			const uint32_t nEliminatedUntilNext(__builtin_ctzll(sieveWord)), candidateIndex((b*64) + nEliminatedUntilNext); // __builtin_ctzll returns the number of leading 0s.
			checkTask.check.factorOffsets[checkTask.check.nCandidates] = candidateIndex;
			checkTask.check.nCandidates++;
			if (checkTask.check.nCandidates == maxCandidatesPerCheckTask) {
				if (_works[workIndex].job.height != _client->currentHeight()) // Low overhead but still often enough
					goto sieveEnd;
				_tasks.push_back(checkTask);
				checkTask.check.nCandidates = 0;
				_works[workIndex].nRemainingCheckTasks++;
			}
			sieveWord &= sieveWord - 1; // Change the candidate's bit from 1 to 0.
		}
	}
	if (_works[workIndex].job.height != _client->currentHeight())
		goto sieveEnd;
	if (checkTask.check.nCandidates > 0) {
		_tasks.push_back(checkTask);
		_works[workIndex].nRemainingCheckTasks++;
	}
	if (sieveIteration + 1 < _parameters.sieveIterations) {
		if (_parameters.threads > 1)
			_tasks.push_front(Task{Task::Type::Sieve, workIndex, .sieve = {sieve.id, sieveIteration + 1}});
		else // Allow mining with 1 Thread without having to wait for all the blocks to be processed.
			_tasks.push_back(Task{Task::Type::Sieve, workIndex, .sieve = {sieve.id, sieveIteration + 1}});
		return; // Sieving still not finished, do not go to sieveEnd.
	}
sieveEnd:
	_tasksDoneInfos.push_back(TaskDoneInfo{Task::Type::Sieve, .empty = {}});
}

// Riecoin uses the Miller-Rabin Test for the PoW, but the Fermat Test is significantly faster and more suitable for the miner.
// n is probably prime if a^(n - 1) ≡ 1 (mod n) for one 0 < a < p or more.
static const mpz_class mpz2(2); // Here, we test with one a = 2.
bool isPrimeFermat(const mpz_class& n) {
	mpz_class r, nm1(n - 1);
	mpz_powm(r.get_mpz_t(), mpz2.get_mpz_t(), nm1.get_mpz_t(), n.get_mpz_t()); // r = 2^(n - 1) % n
	return r == 1;
}

void Miner::_doCheckTask(Task task) {
	const uint16_t workIndex(task.workIndex);
	if (_works[workIndex].job.height != _client->currentHeight()) return;
	std::vector<uint64_t> tupleCounts(_parameters.pattern.size() + 1, 0);
	mpz_class candidateStart, candidate;
	mpz_mul_ui(candidateStart.get_mpz_t(), _primorial.get_mpz_t(), task.check.factorStart);
	candidateStart += _works[workIndex].primorialMultipleStart;
	candidateStart += _primorialOffsets[task.check.offsetId];
	
	for (uint32_t i(0) ; i < task.check.nCandidates ; i++) {
		if (_works[workIndex].job.height != _client->currentHeight()) break;
		candidate = candidateStart + _primorial*task.check.factorOffsets[i];
		
		tupleCounts[0]++;
		if (!isPrimeFermat(candidate)) continue;
		tupleCounts[1]++;
		
		uint32_t primeCount(1), offsetSum(0);
		// Test primality of the other elements of the tuple if candidate + 0 is prime.
		for (std::vector<uint64_t>::size_type i(1) ; i < _parameters.pattern.size() ; i++) {
			offsetSum += _parameters.pattern[i];
			mpz_add_ui(candidate.get_mpz_t(), candidate.get_mpz_t(), _parameters.pattern[i]);
			if (isPrimeFermat(mpz_class(candidate))) {
				primeCount++;
				tupleCounts[primeCount]++;
			}
			else if (_mode == "Pool" && primeCount > 1) {
				int candidatesRemaining(_works[workIndex].job.primeCountTarget - 1 - i);
				if ((primeCount + candidatesRemaining) < _works[workIndex].job.primeCountMin) break; // No chance to be a share anymore
			}
			else break;
		}
		// If tuple long enough or share, submit
		if (primeCount >= _works[workIndex].job.primeCountMin || (_mode == "Search" && primeCount >= _parameters.tupleLengthMin)) {
			const mpz_class basePrime(candidate - offsetSum);
			if (_mode == "Benchmark" || _mode == "Search")
				std::cout << Stats::formattedTime(_statManager.timeSinceStart()) << " " << primeCount;
			else
				std::cout << Stats::formattedClockTimeNow() << " " << primeCount;
			if (_mode == "Pool")
				std::cout << "-share found by worker thread " << threadId << std::endl;
			else {
				std::cout << "-tuple found by worker thread " << threadId << std::endl;
				std::cout << "Base prime: " << basePrime << std::endl;
			}
			Job filledJob(_works[workIndex].job);
			filledJob.result = basePrime;
			filledJob.resultPrimeCount = primeCount;
			filledJob.primorialNumber = _parameters.primorialNumber;
			filledJob.primorialFactor = task.check.factorStart + task.check.factorOffsets[i];
			filledJob.primorialOffset = _parameters.primorialOffsets[task.check.offsetId];
			_client->handleResult(filledJob);
		}
	}
	_statManager.addCounts(tupleCounts);
}

void Miner::_doTasks(const uint16_t id) { // Worker Threads run here until the miner is stopped
	// Thread initialization.
	threadId = id;
	for (int i(0) ; i < _parameters.sieveWorkers ; i++) {
		factorsCache[i] = new uint64_t[factorsCacheSize];
		factorsCacheCounts[i] = new uint64_t[_parameters.sieveIterations];
		for (uint64_t j(0) ; j < _parameters.sieveIterations ; j++)
			factorsCacheCounts[i][j] = 0;
	}
	// Threads are fetching tasks from the queues. The first part of the constellation search is sieving to generate candidates, which is done by the Presieve and Sieve tasks.
	// Once the candidates were generated, they are tested whether they are indeed base primes of constellations using the Fermat Test.
	while (_running) {
		Task task;
		if (!_presieveTasks.try_pop_front(task)) // Presieve Tasks have priority
			task = _tasks.blocking_pop_front();
		
		const auto startTime(std::chrono::steady_clock::now());
		if (task.type == Task::Type::Presieve) {
			_doPresieveTask(task);
			_presieveTime += std::chrono::duration_cast<decltype(_presieveTime)>(std::chrono::steady_clock::now() - startTime);
			_tasksDoneInfos.push_back(TaskDoneInfo{Task::Type::Presieve, .firstPrimeIndex = task.presieve.start});
		}
		if (task.type == Task::Type::Sieve) {
			_doSieveTask(task);
			_sieveTime += std::chrono::duration_cast<decltype(_sieveTime)>(std::chrono::steady_clock::now() - startTime);
			// The Sieve's Task Done Info is created in _doSieveTask
		}
		if (task.type == Task::Type::Check) {
			_doCheckTask(task);
			_verifyTime += std::chrono::duration_cast<decltype(_verifyTime)>(std::chrono::steady_clock::now() - startTime);
			_tasksDoneInfos.push_back(TaskDoneInfo{Task::Type::Check, .workIndex = task.workIndex});
		}
	}
	// Thread clean up.
	for (int i(0) ; i < _parameters.sieveWorkers ; i++) {
		delete factorsCacheCounts[i];
		delete factorsCache[i];
	}
}

void Miner::_manageTasks() {
	Job job; // Block's data (target, blockheader if applicable, ...) from the Client
	_currentWorkIndex = 0;
	uint32_t oldHeight(0);
	while (_running && _client->getJob(job)) {
		if (job.difficulty < _difficultyAtInit - 48. || job.difficulty > _difficultyAtInit + 96.) // Restart to retune parameters.
			_shouldRestart = true;
		if (std::dynamic_pointer_cast<NetworkedClient>(_client) != nullptr) {
			const NetworkInfo networkInfo(std::dynamic_pointer_cast<NetworkedClient>(_client)->info());
			if (!hasAcceptedPatterns(networkInfo.acceptedPatterns)) // Restart if the pattern changed and is no longer compatible with the current one (notably, for the 0.20 fork)
				_shouldRestart = true;
		}
		_presieveTime = _presieveTime.zero();
		_sieveTime = _sieveTime.zero();
		_verifyTime = _verifyTime.zero();
		
		_works[_currentWorkIndex].job = job;
		const bool isNewHeight(oldHeight != _works[_currentWorkIndex].job.height);
		// Notify when the network found a block
		if (isNewHeight && oldHeight != 0) {
			_statManager.newBlock();
			if (_mode == "Benchmark" || _mode == "Search")
				std::cout << Stats::formattedTime(_statManager.timeSinceStart());
			else
				std::cout << Stats::formattedClockTimeNow();
			std::cout << " Block " << job.height << ", average " << FIXED(1) << _statManager.averageBlockTime() << " s, difficulty " << FIXED(3) << job.difficulty << std::endl;
		}
		_works[_currentWorkIndex].primorialMultipleStart = _works[_currentWorkIndex].job.target + _primorial - (_works[_currentWorkIndex].job.target % _primorial);
		// Reset Counts and create Presieve Tasks
		for (auto &sieve : _sieves) {
			for (uint64_t j(0) ; j < _parameters.sieveIterations ; j++)
				sieve.additionalFactorsToEliminateCounts[j] = 0;
		}
		uint64_t nPresieveTasks(_parameters.threads*8ULL);
		int32_t nRemainingNormalPresieveTasks(0), nRemainingAdditionalPresieveTasks(0);
		const uint32_t remainingTasks(_tasks.size());
		const uint64_t primesPerPresieveTask((_nPrimes - _parameters.primorialNumber)/nPresieveTasks + 1ULL);
		for (uint64_t start(_parameters.primorialNumber) ; start < _nPrimes ; start += primesPerPresieveTask) {
			const uint64_t end(std::min(_nPrimes, start + primesPerPresieveTask));
			_presieveTasks.push_back(Task{Task::Type::Presieve, _currentWorkIndex, .presieve = {start, end}});
			_tasks.push_front(Task{Task::Type::Dummy, _currentWorkIndex, {}}); // To ensure a thread wakes up to grab the mod work.
			if (start < _primesIndexThreshold) nRemainingNormalPresieveTasks++;
			else nRemainingAdditionalPresieveTasks++;
		}
		
		while (nRemainingNormalPresieveTasks > 0) {
			const TaskDoneInfo taskDoneInfo(_tasksDoneInfos.blocking_pop_front());
			if (!_running) return; // Can happen if stopThreads is called while this Thread is stuck in this blocking_pop_front().
			if (taskDoneInfo.type == Task::Type::Presieve) {
				if (taskDoneInfo.firstPrimeIndex < _primesIndexThreshold) nRemainingNormalPresieveTasks--;
				else nRemainingAdditionalPresieveTasks--;
			}
			else if (taskDoneInfo.type == Task::Type::Check) _works[taskDoneInfo.workIndex].nRemainingCheckTasks--;
			else ERRORMSG("Unexpected Sieve Task done during Presieving");
		}
		assert(_works[_currentWorkIndex].nRemainingCheckTasks == 0);
		
		// Create Sieve Tasks
		for (std::vector<Sieve>::size_type i(0) ; i < _sieves.size() ; i++) {
			_sieves[i].presieveLock.lock();
			_tasks.push_front(Task{Task::Type::Sieve, _currentWorkIndex, .sieve = {static_cast<uint32_t>(i), 0}});
		}
		
		int nRemainingSieves(_parameters.sieveWorkers);
		while (nRemainingAdditionalPresieveTasks > 0) {
			const TaskDoneInfo taskDoneInfo(_tasksDoneInfos.blocking_pop_front());
			if (!_running) return;
			if (taskDoneInfo.type == Task::Type::Presieve) nRemainingAdditionalPresieveTasks--;
			else if (taskDoneInfo.type == Task::Type::Sieve) nRemainingSieves--;
			else _works[taskDoneInfo.workIndex].nRemainingCheckTasks--;
		}
		for (auto &sieve : _sieves) sieve.presieveLock.unlock();
		
		uint32_t nRemainingTasksMin(std::min(remainingTasks, _tasks.size()));
		while (nRemainingSieves > 0) {
			const TaskDoneInfo taskDoneInfo(_tasksDoneInfos.blocking_pop_front());
			if (!_running) return;
			if (taskDoneInfo.type == Task::Type::Sieve) nRemainingSieves--;
			else if (taskDoneInfo.type == Task::Type::Check) _works[taskDoneInfo.workIndex].nRemainingCheckTasks--;
			else ERRORMSG("Unexpected Presieve Task done during Sieving");
			nRemainingTasksMin = std::min(nRemainingTasksMin, _tasks.size());
		}
		
		// Adjust the Remaining Tasks Threshold
		if (_works[_currentWorkIndex].job.height == _client->currentHeight() && !isNewHeight) {
			DBG(std::cout << "Min work outstanding during sieving: " << nRemainingTasksMin << std::endl;);
			if (remainingTasks > _nRemainingCheckTasksThreshold - _parameters.threads*2) {
				// If we are acheiving our work target, then adjust it towards the amount
				// required to maintain a healthy minimum work queue length.
				if (nRemainingTasksMin == 0) // Need more, but don't know how much, try adding some.
					_nRemainingCheckTasksThreshold += 4*_parameters.threads*_parameters.sieveWorkers;
				else { // Adjust towards target of min work = 4*threads.
					const uint32_t targetMaxWork((_nRemainingCheckTasksThreshold - nRemainingTasksMin) + 8*_parameters.threads);
					_nRemainingCheckTasksThreshold = (_nRemainingCheckTasksThreshold + targetMaxWork)/2;
				}
			}
			else if (nRemainingTasksMin > 4u*_parameters.threads) { // Didn't make the target, but also didn't run out of work. Can still adjust target.
				const uint32_t targetMaxWork((remainingTasks - nRemainingTasksMin) + 10*_parameters.threads);
				_nRemainingCheckTasksThreshold = (_nRemainingCheckTasksThreshold + targetMaxWork)/2;
			}
			else if (nRemainingTasksMin == 0 && remainingTasks > 0) {
				static int allowedFails(5);
				if (--allowedFails == 0) { // Warn possible CPU Underuse
					allowedFails = 5;
					DBG(std::cout << "Unable to generate enough verification work to keep threads busy." << std::endl;);
				}
			}
			_nRemainingCheckTasksThreshold = std::min(_nRemainingCheckTasksThreshold, _tasksDoneInfos.size() - 9*_parameters.threads);
			DBG(std::cout << "Work target before starting next block now: " << _nRemainingCheckTasksThreshold << std::endl;);
		}
		
		oldHeight = _works[_currentWorkIndex].job.height;
		
		while (_works[_currentWorkIndex].nRemainingCheckTasks > _nRemainingCheckTasksThreshold) {
			const TaskDoneInfo taskDoneInfo(_tasksDoneInfos.blocking_pop_front());
			if (!_running) return;
			if (taskDoneInfo.type == Task::Type::Check) _works[taskDoneInfo.workIndex].nRemainingCheckTasks--;
			else ERRORMSG("Expected Check Task done");
		}
		_currentWorkIndex = (_currentWorkIndex + 1) % nWorks;
		while (_works[_currentWorkIndex].nRemainingCheckTasks > 0) {
			const TaskDoneInfo taskDoneInfo(_tasksDoneInfos.blocking_pop_front());
			if (!_running) return;
			if (taskDoneInfo.type == Task::Type::Check) _works[taskDoneInfo.workIndex].nRemainingCheckTasks--;
			else ERRORMSG("Expected Check Task done 2");
		}
		
		DBG(std::cout << "Job Timing: " << _presieveTime.count() << "/" << _sieveTime.count() << "/" << _verifyTime.count() << ", tasks: " << _works[0].nRemainingCheckTasks << ", " << _works[1].nRemainingCheckTasks << std::endl;);
	}
}

void Miner::_suggestLessMemoryIntensiveOptions(const uint64_t suggestedPrimeTableLimit, const uint16_t suggestedSieveWorkers) const {
	std::cout << "You don't have enough available memory to run rieMiner with the current options." << std::endl;
	std::cout << "Try to use the following options in the " << confPath << " configuration file and retry:" << std::endl;
	std::cout << "PrimeTableLimit = " << suggestedPrimeTableLimit << std::endl;
	std::cout << "SieveWorkers = " << suggestedSieveWorkers << std::endl;
}

bool Miner::hasAcceptedPatterns(const std::vector<std::vector<uint64_t>> &acceptedPatterns) const {
	for (const auto &acceptedPattern : acceptedPatterns) {
		bool compatible(true);
		for (uint16_t i(0) ; i < acceptedPattern.size() ; i++) {
			if (i >= _parameters.pattern.size() ? true : acceptedPattern[i] != _parameters.pattern[i]) {
				compatible = false;
				break;
			}
		}
		if (compatible)
			return true;
	}
	return false;
}

void Miner::printStats() const {
	Stats statsRecent(_statManager.stats(false)), statsSinceStart(_statManager.stats(true));
	if (_mode == "Benchmark" || _mode == "Search") {
		statsRecent = statsSinceStart;
		std::cout << Stats::formattedTime(_statManager.timeSinceStart());
	}
	else
		std::cout << Stats::formattedClockTimeNow();
	std::cout << " " << FIXED(2) << statsRecent.cps() << " c/s, r " << statsRecent.r();
	if (_mode != "Pool") {
		std::cout << " ; (1-" << _parameters.pattern.size() << "t) = " << statsSinceStart.formattedCounts(1);
		if (statsRecent.count(1) >= 10)
			std::cout << " | " << Stats::formattedDuration(statsRecent.estimatedAverageTimeToFindBlock(_works[_currentWorkIndex].job.primeCountTarget));
	}
	else {
		std::dynamic_pointer_cast<StratumClient>(_client)->printSharesStats();
		if (statsRecent.count(1) >= 10)
			std::cout << " | " << 86400.*(50./static_cast<double>(1 << _client->currentHeight()/840000))/statsRecent.estimatedAverageTimeToFindBlock(_works[_currentWorkIndex].job.primeCountTarget) << " RIC/d";
	}
	std::cout << std::endl;
}
bool Miner::benchmarkFinishedTimeOut(const double benchmarkTimeLimit) const {
	const Stats stats(_statManager.stats(true));
	return benchmarkTimeLimit > 0. && stats.duration() >= benchmarkTimeLimit;
}
bool Miner::benchmarkFinishedEnoughPrimes(const uint64_t benchmarkPrimeCountLimit) const {
	const Stats stats(_statManager.stats(true));
	return benchmarkPrimeCountLimit > 0 && stats.count(1) >= benchmarkPrimeCountLimit;
}
void Miner::printBenchmarkResults() const {
	Stats stats(_statManager.stats(true));
	std::cout << "Benchmark finished after " << stats.duration() << " s." << std::endl;
	std::cout << FIXED(6) << stats.cps() << " candidates/s, ratio " << stats.r() << " -> " << 86400./stats.estimatedAverageTimeToFindBlock(_works[_currentWorkIndex].job.primeCountTarget) << " block(s)/day" << std::endl;
}
void Miner::printTupleStats() const {
	Stats stats(_statManager.stats(true));
	std::cout << "Tuples found: " << stats.formattedCounts() << " in " << FIXED(6) << stats.duration() << " s" << std::endl;
	std::cout << "Tuple rates : " << stats.formattedRates() << std::endl;
	std::cout << "Tuple ratios: " << stats.formattedRatios() << std::endl;
}
