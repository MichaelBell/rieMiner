#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>

#include "prime-gmp.h"

#include "primetestcxt.h"

typedef uint32_t uint;
typedef uint64_t ulong;

#define MAX_JOB_SIZE MAX_JOB_SIZE_PRIME
#define MAX_BLOCK_SIZE 64

template<uint N_Size>
__device__
void squareSimple(uint* P, const uint* R) {
	const uint highbit = ((uint)1) << 31;

	uint T[(N_Size - 1) * 2];

	{
		uint cy = 0;
		for (int i = 0; i < N_Size - 1; ++i)
		{
			ulong p = ulong(R[i + 1]) * ulong(R[0]) + cy;
			T[i] = uint(p);
			cy = uint(p >> 32);
		}
		T[N_Size - 1] = cy;
	}

#pragma unroll 1
	for (int j = 2; j < N_Size; ++j)
	{
		uint cy = 0;
		for (int i = j; i < N_Size; ++i)
		{
			ulong p = ulong(R[i]) * ulong(R[j - 1]);
			p += cy;
			p += T[i + j - 2];
			T[i + j - 2] = uint(p);
			cy = uint(p >> 32);
		}
		T[N_Size + j - 2] = cy;
	}

	// Better not to include this into the next loop as doing it first
	// avoids latency stalls.
	for (int i = 0; i < N_Size; ++i)
	{
		P[2 * i] = R[i] * R[i];
		P[2 * i + 1] = __umulhi(R[i], R[i]);
	}

	uint cy = 0;
	for (int i = 0; i < N_Size - 1; ++i)
	{
		uint t = T[2 * i] & highbit;
		ulong a = ulong(P[2 * i + 1]) + cy;
		a += T[2 * i] << 1;
		P[2 * i + 1] = uint(a);
		cy = (t >> 31) + uint(a >> 32);

		t = T[2 * i + 1] & highbit;
		a = ulong(P[2 * i + 2]) + cy;
		a += T[2 * i + 1] << 1;
		P[2 * i + 2] = uint(a);
		cy = (t >> 31) + uint(a >> 32);
	}
	P[2 * N_Size - 1] += cy;
}

template<uint N_Size>
__device__
uint bigAdd(uint* R, const uint* A, const uint* B)
{
        uint cy = 0;
        for (int i = 0; i < N_Size; ++i)
        {
                ulong a = (ulong)A[i] + (ulong)B[i];
                a += cy;
                R[i] = (uint)a;
                cy = (uint)(a >> 32);
        }
        return cy;
}

template<uint N_Size>
__device__
int32_t bigSub(uint* R, const uint* A, const uint* B)
{
        int32_t cy = 0;
        for (int i = 0; i < N_Size; ++i)
        {
                int64_t a = (int64_t)A[i] - (int64_t)B[i];
                a += cy;
                R[i] = (uint)a;
                cy = (int32_t)(a >> 32);
        }
        return cy;
}

template<uint N_Size>
__device__
bool lessThan(const uint* A, const uint* B)
{
        for (int i = N_Size - 1; i >= 0; --i)
        {
                if (A[i] < B[i]) return true;
                if (A[i] > B[i]) return false;
        }
        return false;
}

template<uint N_Size>
__device__
void toom2SquareFull(uint* P, const uint* R)
{
        constexpr uint s = N_Size >> 1;
        constexpr uint n = N_Size - s;  // n==s or n==s+1

        const uint* a0 = R;
        const uint* a1 = R + n;
        uint* asm1 = P;

        if (s == n)
        {
                if (lessThan<n>(a0, a1))
                        bigSub<n>(asm1, a1, a0);
                else
                        bigSub<n>(asm1, a0, a1);
        }
        else
        {
                if (a0[s] == 0 && lessThan<s>(a0, a1))
                {
                        bigSub<s>(asm1, a1, a0);
                        asm1[s] = 0;
                }
                else
                {
                        asm1[s] = a0[s] + bigSub<s>(asm1, a0, a1);
                }
                P[4*n-2] = 0;
                P[4*n-1] = 0;
        }

        uint* v0 = P;
        uint vm1[N_Size + 1];
        uint* vinf = P + 2 * n;

        squareSimple<n>(vm1, asm1);
        squareSimple<s>(vinf, a1);
        squareSimple<n>(v0, a0);
        //toom2SquareHalf(vm1, asm1, n);
        //toom2SquareHalf(vinf, a1, s);
        //toom2SquareHalf(v0, a0, n);

        int32_t cy  = bigAdd<n>(P + 2*n, v0 + n, vinf);
        uint cy2 = cy + bigAdd<n>(P + n, P + 2*n, v0);
        cy += bigAdd<n>(P + 2*n, P + 2*n, vinf + n);
        cy += bigSub<2*n>(P + n, P + n, vm1);

        for (int i = 2*n; cy2 > 0; ++i)
        {
                ulong a = (ulong)P[i] + cy2;
                P[i] = (uint)a;
                cy2 = (uint)(a >> 32);
        }
        for (int i = 3*n; cy != 0; ++i)
        {
                int64_t a = (int64_t)P[i] + cy;
                P[i] = (uint)a;
                cy = (int32_t)(a >> 32);
        }
}

template<uint N_Size>
__global__
void fermat_test(const uint *M_in, const uint *Mi_in, const uint *R_in, uint *is_prime) {

	uint R[N_Size];
	uint M[N_Size];

	{
		// Get the index of the current element to be processed
		const int offset = (blockDim.x*blockIdx.x + threadIdx.x) * N_Size;

		for (int i = 0; i < N_Size; ++i)
		{
			M[i] = M_in[offset + i];
			R[i] = R_in[offset + i];
		}
	}

	const uint shift = __clz(M[N_Size - 1]);
	const uint highbit = ((uint)1) << 31;
	uint startbit;
	int en = N_Size;

        if (shift < 24)
        {
                startbit = highbit >> (shift + 8);
        }
        else
        {
                startbit = highbit >> (shift - 24);
                en--;
        }

	const uint mi = Mi_in[blockDim.x*blockIdx.x + threadIdx.x];

#pragma unroll 1
	while (en-- > 0)
	{
		uint bit = startbit;
		startbit = highbit;
		uint E = M[en];
		if (en == 0) E--;

		do
		{
			{
				uint P[N_Size * 2 + 1];
				//mpn_sqr(pp, rp, mn);
				squareSimple<N_Size>(P, R);
				//toom2SquareFull<N_Size>(P, R);

				//if (mpn_redc_1(rp, pp, mp, mn, mi) != 0) 
				//  mpn_sub_n(rp, rp, mshifted, n);
#pragma unroll 1
				for (int j = 0; j < N_Size; ++j)
				{
					uint cy = 0;
					uint v = P[j] * mi;
					for (int i = 0; i < N_Size; ++i)
					{
						ulong p = ulong(M[i]) * ulong(v) + cy;
						p += P[i + j];
						P[i + j] = uint(p);
						cy = uint(p >> 32);
					}
					R[j] = cy;
				}

				{
					uint cy = 0;
					for (int i = 0; i < N_Size; ++i)
					{
						ulong a = ulong(R[i]) + cy;
						a += P[i + N_Size];
						R[i] = uint(a);
						cy = uint(a >> 32);
					}

					if (cy != 0)
					{
						int32_t borrow = 0;
						uint last_shifted = 0;
						for (int i = 0; i < N_Size; ++i)
						{
							int64_t a = R[i];
							uint b = (M[i] << shift) | last_shifted;
							last_shifted = M[i] >> (32 - shift);
							a = a - int64_t(b) + borrow;
							R[i] = uint(a);
							borrow = int32_t(a >> 32);
						}
					}
				}
			}

			if (E & bit)
			{
				//mp_limb_t carry = mpn_lshift(rp, rp, mn, 1);
				uint carry = 0;
				for (int i = 0; i < N_Size; ++i)
				{
					uint t = R[i] & highbit;
					R[i] <<= 1;
					R[i] |= carry;
					carry = t >> 31;
				}
				while (carry)
				{
					//carry -= mpn_sub_n(rp, rp, mshifted, mn);
					int32_t borrow = 0;
					uint last_shifted = 0;
					for (int i = 0; i < N_Size; ++i)
					{
						int64_t a = R[i];
						uint b = (M[i] << shift) | last_shifted;
						last_shifted = M[i] >> (32 - shift);
						a = a - int64_t(b) + borrow;
						R[i] = uint(a);
						borrow = int32_t(a >> 32);
					}
					carry += borrow;
				}
			}
			bit >>= 1;
		} while (bit > 0);

	}

	// DeREDCify - necessary as rp can have a large
	//             multiple of m in it (although I'm not 100% sure
	//             why it can't after this redc!)
	{
		uint T[N_Size * 2];
		for (int i = 0; i < N_Size; ++i)
		{
			T[i] = R[i];
			T[N_Size + i] = 0;
		}

		// MPN_REDC_1(rp, tp, mp, mn, mi);
#pragma unroll 1
		for (int j = 0; j < N_Size; ++j)
		{
			uint cy = 0;
			uint v = T[j] * mi;
			for (int i = 0; i < N_Size; ++i)
			{
				ulong p = ulong(M[i]) * ulong(v) + cy;
				p += T[i + j];
				T[i + j] = uint(p);
				cy = uint(p >> 32);
			}
			R[j] = cy;
		}

		{
			uint cy = 0;
			for (int i = 0; i < N_Size; ++i)
			{
				ulong a = ulong(R[i]) + cy;
				a += T[i + N_Size];
				R[i] = uint(a);
				cy = uint(a >> 32);
			}

			if (cy != 0)
			{
				int32_t borrow = 0;
				uint last_shifted = 0;
				for (int i = 0; i < N_Size; ++i)
				{
					int64_t a = R[i];
					uint b = (M[i] << shift) | last_shifted;
					last_shifted = M[i] >> (32 - shift);
					a = a - int64_t(b) + borrow;
					R[i] = uint(a);
					borrow = int32_t(a >> 32);
				}
			}
		}
	}

	bool result = true;
	if (R[N_Size - 1] != 0)
	{
		// Compare to m+1
		uint cy = 1;
		for (int i = 0; i < N_Size && result; ++i)
		{
			uint a = M[i] + cy;
			cy = a < M[i];
			if (R[i] != a) result = false;
		}
	}
	else
	{
		// Compare to 1
		result = R[0] == 1;
		for (int i = 1; i < N_Size && result; ++i)
		{
			if (R[i] != 0) result = false;
		}
	}

	is_prime[blockDim.x*blockIdx.x + threadIdx.x] = result;
}

#define DEBUG 0

#define MAX_SOURCE_SIZE (0x100000)

const unsigned char  binvert_limb_table[128] = {
	0x01, 0xAB, 0xCD, 0xB7, 0x39, 0xA3, 0xC5, 0xEF,
	0xF1, 0x1B, 0x3D, 0xA7, 0x29, 0x13, 0x35, 0xDF,
	0xE1, 0x8B, 0xAD, 0x97, 0x19, 0x83, 0xA5, 0xCF,
	0xD1, 0xFB, 0x1D, 0x87, 0x09, 0xF3, 0x15, 0xBF,
	0xC1, 0x6B, 0x8D, 0x77, 0xF9, 0x63, 0x85, 0xAF,
	0xB1, 0xDB, 0xFD, 0x67, 0xE9, 0xD3, 0xF5, 0x9F,
	0xA1, 0x4B, 0x6D, 0x57, 0xD9, 0x43, 0x65, 0x8F,
	0x91, 0xBB, 0xDD, 0x47, 0xC9, 0xB3, 0xD5, 0x7F,
	0x81, 0x2B, 0x4D, 0x37, 0xB9, 0x23, 0x45, 0x6F,
	0x71, 0x9B, 0xBD, 0x27, 0xA9, 0x93, 0xB5, 0x5F,
	0x61, 0x0B, 0x2D, 0x17, 0x99, 0x03, 0x25, 0x4F,
	0x51, 0x7B, 0x9D, 0x07, 0x89, 0x73, 0x95, 0x3F,
	0x41, 0xEB, 0x0D, 0xF7, 0x79, 0xE3, 0x05, 0x2F,
	0x31, 0x5B, 0x7D, 0xE7, 0x69, 0x53, 0x75, 0x1F,
	0x21, 0xCB, 0xED, 0xD7, 0x59, 0xC3, 0xE5, 0x0F,
	0x11, 0x3B, 0x5D, 0xC7, 0x49, 0x33, 0x55, 0xFF
};

#define binvert_limb(inv,n)                                             \
  do {                                                                  \
    mp_limb_t  __n = (n);                                               \
    mp_limb_t  __inv;                                                   \
    assert ((__n & 1) == 1);                                            \
                                                                        \
    __inv = binvert_limb_table[(__n/2) & 0x7F]; /*  8 */                \
    if (GMP_LIMB_BITS > 8)   __inv = 2 * __inv - __inv * __inv * __n;   \
    if (GMP_LIMB_BITS > 16)  __inv = 2 * __inv - __inv * __inv * __n;   \
    if (GMP_LIMB_BITS > 32)  __inv = 2 * __inv - __inv * __inv * __n;   \
                                                                        \
    assert ((__inv * __n) == 1);                        \
    (inv) = __inv;                                      \
  } while (0)

static void setup_fermat(int N_Size, int num, const mp_limb_t* M, mp_limb_t* MI, mp_limb_t* R)
{
	assert(N_Size <= MAX_N_SIZE);
	for (int j = 0; j < num; ++j)
	{
		mp_size_t mn = N_Size;
		mp_limb_t mshifted[MAX_N_SIZE];
		mp_srcptr mp;
		mp_ptr rp;
		struct gmp_div_inverse minv;

		// REDCify: r = B^n * 2 % M
		mp = &M[j*N_Size];
		rp = &R[j*N_Size];
		mpn_div_qr_invert(&minv, mp, mn);

		if (minv.shift > 0)
		{
			mpn_lshift(mshifted, mp, mn, minv.shift);
			mp = mshifted;
		}

		for (size_t i = 0; i < mn + 4; ++i) rp[i] = 0;
		rp[mn + 4] = 1 << minv.shift;
		mpn_div_r_preinv_ns(rp, mn + 5, mp, mn, &minv);

		if (minv.shift > 0)
		{
			mpn_rshift(rp, rp, mn, minv.shift);
			mp = &M[j*N_Size];
		}

		mp_limb_t mi;
		binvert_limb(mi, mp[0]);
		MI[j] = -mi;
	}
}

#if DEBUG
#define DPRINTF(fmt, args...) do { printf("line %d: " fmt, __LINE__, ##args); fflush(stdout); } while(0)
#else
#define DPRINTF(fmt, ...) do { } while(0)
#endif

PrimeTestCxt* primeTestInit()
{
	cudaError_t cudaStatus;

	PrimeTestCxt* cxt = new PrimeTestCxt;

	int device;
	CUresult cuResult;
	cuResult = cuInit(0);
	if (cuResult != CUDA_SUCCESS) {
		printf("cuInit failed!");
		abort();
	}
	cuResult = cuDeviceGet(&device, 0);
	if (cuResult != CUDA_SUCCESS) {
		printf("cuDeviceGet failed!");
		abort();
	}
	cuResult = cuCtxCreate(&cxt->cudaCxt, CU_CTX_SCHED_BLOCKING_SYNC, device);
	if (cuResult != CUDA_SUCCESS) {
		printf("cuCtxCreate failed!");
		abort();
	}

	// Create memory buffers on the device for each vector 
	cudaStatus = cudaMalloc((void**)&cxt->m_mem_obj, MAX_JOB_SIZE * MAX_N_SIZE * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->mi_mem_obj, max(MAX_JOB_SIZE_PRIME, MAX_JOB_SIZE_MOD) * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->r_mem_obj, max(MAX_JOB_SIZE_PRIME * MAX_N_SIZE + 5, MAX_JOB_SIZE_MOD * 6) * sizeof(uint));
	cudaStatus = cudaMalloc((void**)&cxt->is_prime_mem_obj, max(MAX_JOB_SIZE_PRIME, MAX_JOB_SIZE_MOD * 2) * sizeof(uint));

	cudaStatus = cudaEventCreateWithFlags(&cxt->cudaEvent, cudaEventBlockingSync);

	// Create buffers on host
	cudaMallocHost((void**)&cxt->R, sizeof(uint)*max(MAX_N_SIZE*MAX_JOB_SIZE + 5, MAX_JOB_SIZE_MOD * 6));
	cudaMallocHost((void**)&cxt->MI, sizeof(uint)*max(MAX_JOB_SIZE_PRIME, MAX_JOB_SIZE_MOD));
	cudaMallocHost((void**)&cxt->is_prime, sizeof(uint)*max(MAX_JOB_SIZE_PRIME, MAX_JOB_SIZE_MOD * 2));

	cuCtxPopCurrent(NULL);

	return cxt;
}

void primeTest(PrimeTestCxt* cxt, int N_Size, int listSize, const uint* M, uint* is_prime, void (*workFn)(void*), void* workCxt)
{
	std::lock_guard<std::mutex> lock(cxt->cudaMutex);
	cudaError_t cudaStatus;
	CUresult cuResult;

	cuResult = cuCtxPushCurrent(cxt->cudaCxt);
	if (cuResult != CUDA_SUCCESS) {
		printf("cuCtxPushCurrent failed!");
		abort();
	}

	if (N_Size < 8 || N_Size > MAX_N_SIZE)
	{
		printf("N Size out of bounds\n");
		abort();
	}

	int nextJobSize = min(MAX_JOB_SIZE, listSize);
	int jobSize = 0;
	int lastJobSize = 0;

	if (nextJobSize > 0)
	{
		setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);
	}

	while (nextJobSize > 0)
	{
		lastJobSize = jobSize;
		jobSize = nextJobSize;
		listSize -= jobSize;
		nextJobSize = min(MAX_JOB_SIZE, listSize);

		// Copy the lists A and B to their respective memory buffers
		cudaStatus = cudaMemcpyAsync(cxt->mi_mem_obj, cxt->MI, jobSize * sizeof(uint), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpyAsync(cxt->r_mem_obj, cxt->R, jobSize * N_Size * sizeof(uint), cudaMemcpyHostToDevice);
		cudaEventRecord(cxt->cudaEvent);
		cudaStatus = cudaMemcpyAsync(cxt->m_mem_obj, M, jobSize * N_Size * sizeof(uint), cudaMemcpyHostToDevice);

		int blockSize = 1;
		int numBlocks = jobSize;
		while (blockSize < MAX_BLOCK_SIZE && ((numBlocks & 1) == 0))
		{
			numBlocks >>= 1;
			blockSize <<= 1;
		}

		DPRINTF("before execution\n");
#define TEST(N) case N: fermat_test<N> << <numBlocks, blockSize >> >(cxt->m_mem_obj, cxt->mi_mem_obj, cxt->r_mem_obj, cxt->is_prime_mem_obj); break
		switch (N_Size)
		{
			TEST(8);
			TEST(9);
			TEST(10);
			TEST(11);
			TEST(12);
			TEST(13);
			TEST(14);
			TEST(15);
			TEST(16);
			TEST(17);
			TEST(18);
			TEST(19);
			TEST(20);
			TEST(21);
			TEST(22);
			TEST(23);
			TEST(24);
			TEST(25);
			TEST(26);
			TEST(27);
			TEST(28);
			TEST(29);
			TEST(30);
			TEST(31);
			TEST(32);
			TEST(33);
			TEST(34);
			TEST(35);
			TEST(36);
			TEST(37);
			TEST(38);
			TEST(39);
			TEST(40);
			TEST(41);
			TEST(42);
			TEST(43);
			TEST(44);
			TEST(45);
			TEST(46);
			TEST(47);
			TEST(48);
#if 0
			TEST(49);
			TEST(50);
			TEST(51);
			TEST(52);
			TEST(53);
			TEST(54);
			TEST(55);
			TEST(56);
			TEST(57);
			TEST(58);
			TEST(59);
			TEST(60);
			TEST(61);
			TEST(62);
			TEST(63);
			TEST(64);
			TEST(65);
			TEST(66);
			TEST(67);
			TEST(68);
			TEST(69);
			TEST(70);
			TEST(71);
			TEST(72);
			TEST(73);
			TEST(74);
			TEST(75);
			TEST(76);
			TEST(77);
			TEST(78);
			TEST(79);
			TEST(80);
			TEST(81);
			TEST(82);
			TEST(83);
			TEST(84);
			TEST(85);
			TEST(86);
			TEST(87);
			TEST(88);
			TEST(89);
			TEST(90);
			TEST(91);
			TEST(92);
			TEST(93);
			TEST(94);
			TEST(95);
			TEST(96);
			TEST(97);
			TEST(98);
			TEST(99);
			TEST(100);
			TEST(101);
			TEST(102);
			TEST(103);
			TEST(104);
			TEST(105);
			TEST(106);
			TEST(107);
			TEST(108);
			TEST(109);
			TEST(110);
			TEST(111);
			TEST(112);
			TEST(113);
			TEST(114);
			TEST(115);
			TEST(116);
			TEST(117);
			TEST(118);
			TEST(119);
			TEST(120);
			TEST(121);
			TEST(122);
			TEST(123);
			TEST(124);
			TEST(125);
			TEST(126);
			TEST(127);
#endif
		default: abort();
		}
		
#if 1
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
		}
#endif
		if (workFn)
		{
			workFn(workCxt);
			workFn = NULL;
		}

		cudaEventSynchronize(cxt->cudaEvent);
		if (cudaStatus != cudaSuccess) {
			printf("Sync failed: %s\n", cudaGetErrorString(cudaStatus));
			break;
		}

		if (lastJobSize > 0)
		{
			memcpy(is_prime, cxt->is_prime, lastJobSize * sizeof(uint));
			is_prime += lastJobSize;
		}

		if (nextJobSize > 0)
		{
			M += jobSize*N_Size;
			setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);

			cudaStatus = cudaMemcpyAsync(cxt->is_prime, cxt->is_prime_mem_obj, jobSize * sizeof(uint), cudaMemcpyDeviceToHost);
		}
		else
		{
			cuResult = cuCtxSynchronize();
			if (cuResult != CUDA_SUCCESS) {
				printf("cuCtxSynchronize failed!");
			}

			cudaStatus = cudaMemcpy(is_prime, cxt->is_prime_mem_obj, jobSize * sizeof(uint), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				printf("Final memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
			}
		}
	}

	cuResult = cuCtxPopCurrent(NULL);
	if (cuResult != CUDA_SUCCESS) {
		printf("cuCtxPopCurrent failed!");
		abort();
	}
}

void primeTestTerm(PrimeTestCxt* cxt)
{
	cuCtxPushCurrent(cxt->cudaCxt);
	cudaFree(cxt->mi_mem_obj);
	cudaFree(cxt->m_mem_obj);
	cudaFree(cxt->r_mem_obj);
	cudaFree(cxt->is_prime_mem_obj);

	cudaFreeHost(cxt->R);
	cudaFreeHost(cxt->MI);
	cudaFreeHost(cxt->is_prime);
	delete cxt;

	cuCtxDestroy(cxt->cudaCxt);

	cudaDeviceReset();
}
