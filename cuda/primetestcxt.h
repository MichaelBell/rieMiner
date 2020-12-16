#pragma once

#define MAX_N_SIZE 48
#define MAX_JOB_SIZE_PRIME 512
#define MAX_JOB_SIZE_MOD 65536

typedef struct PrimeTestCxt
{
	uint32_t* m_mem_obj;
	uint32_t* mi_mem_obj;
	uint32_t* r_mem_obj;
	uint32_t* is_prime_mem_obj;

	CUcontext cudaCxt;
	cudaEvent_t cudaEvent;

	std::mutex cudaMutex;

	uint32_t *R;
	uint32_t *MI;
	uint32_t* is_prime;
} PrimeTestCxt;
