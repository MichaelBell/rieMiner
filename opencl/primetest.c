#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#include "prime-gmp.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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

#define MAX_N_SIZE 96

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

		for (size_t i = 0; i < mn; ++i) rp[i] = 0;
		rp[mn] = 1 << minv.shift;
		mpn_div_r_preinv_ns(rp, mn + 1, mp, mn, &minv);

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

typedef struct PrimeTestCxt
{
	cl_device_id device_id;
	cl_command_queue command_queue;

	cl_mem m_mem_obj;
	cl_mem mi_mem_obj;
	cl_mem r_mem_obj;
	cl_mem is_prime_mem_obj;

	cl_context context;
	cl_program program;
	cl_kernel kernel;

	int N_Size;
	cl_uint *R;
	cl_uint *MI;
} PrimeTestCxt;

#define MAX_JOB_SIZE 1024

PrimeTestCxt* primeTestInit()
{
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("opencl/primetest.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load CL program.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	DPRINTF("kernel loading done\n");

	PrimeTestCxt* cxt = (PrimeTestCxt*)malloc(sizeof(PrimeTestCxt));
	cxt->device_id = NULL;

	// Get platform and device information
	cl_uint ret_num_platforms;

	cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1,
						 &cxt->device_id, NULL);

	// Create an OpenCL context
	cxt->context = clCreateContext(NULL, 1, &cxt->device_id, NULL, NULL, &ret);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	// Create a command queue
	cxt->command_queue = clCreateCommandQueue(cxt->context, cxt->device_id, 0, &ret);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	// Create memory buffers on the device for each vector 
	cxt->m_mem_obj = clCreateBuffer(cxt->context, CL_MEM_READ_ONLY,
									  MAX_JOB_SIZE * MAX_N_SIZE * sizeof(cl_uint), NULL, &ret);
	cxt->mi_mem_obj = clCreateBuffer(cxt->context, CL_MEM_READ_ONLY,
									 MAX_JOB_SIZE * sizeof(cl_uint), NULL, &ret);
	cxt->r_mem_obj = clCreateBuffer(cxt->context, CL_MEM_READ_ONLY,
									MAX_JOB_SIZE * MAX_N_SIZE * sizeof(cl_uint), NULL, &ret);
	cxt->is_prime_mem_obj = clCreateBuffer(cxt->context, CL_MEM_WRITE_ONLY,
										   MAX_JOB_SIZE * sizeof(cl_uint), NULL, &ret);

	// Create a program from the kernel source
	cxt->program = clCreateProgramWithSource(cxt->context, 1,
		(const char **)&source_str, (const size_t *)&source_size, &ret);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	cxt->R = (cl_uint*)malloc(sizeof(cl_uint)*(MAX_N_SIZE*MAX_JOB_SIZE + 1));
	cxt->MI = (cl_uint*)malloc(sizeof(cl_uint)*MAX_JOB_SIZE);
	cxt->N_Size = 0;

	return cxt;
}

static void primeTestBuild(PrimeTestCxt* cxt, int N_Size)
{
	if (cxt->N_Size != 0)
	{
		clReleaseKernel(cxt->kernel);
	}
	if (N_Size > MAX_N_SIZE)
	{
		printf("N Size above maximum\n");
		abort();
	}

	// Build the program
	char options[1024];
	sprintf(options, "-DN_Size=%d", N_Size);
	cl_int ret = clBuildProgram(cxt->program, 1, &cxt->device_id, options, NULL, NULL);

	if (ret != CL_SUCCESS)
	{
		printf("Build program returned %d\n", ret);

		char str[32768];
		size_t str_len;
		clGetProgramBuildInfo(cxt->program, cxt->device_id, CL_PROGRAM_BUILD_LOG, sizeof(str), str, &str_len);
		str[min(str_len, 1023)] = 0;
		printf("Build Log: %s\n", str);
		abort();
	}

	// Create the OpenCL kernel
	cxt->kernel = clCreateKernel(cxt->program, "fermat_test", &ret);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(cxt->kernel, 0, sizeof(cl_mem), (void *)&cxt->m_mem_obj);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(cxt->kernel, 1, sizeof(cl_mem), (void *)&cxt->mi_mem_obj);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(cxt->kernel, 2, sizeof(cl_mem), (void *)&cxt->r_mem_obj);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	ret = clSetKernelArg(cxt->kernel, 3, sizeof(cl_mem), (void *)&cxt->is_prime_mem_obj);
	DPRINTF("ret at %d is %d\n", __LINE__, ret);

	cxt->N_Size = N_Size;
}

void primeTest(PrimeTestCxt* cxt, int N_Size, int listSize, const cl_uint* M, cl_uint* is_prime)
{
	if (N_Size != cxt->N_Size)
	{
		primeTestBuild(cxt, N_Size);
	}

	int nextJobSize = min(MAX_JOB_SIZE, listSize);

	if (nextJobSize > 0)
	{
		setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);
	}

	cl_event complete_events[2];
	int firstJob = 1;
	while (nextJobSize > 0)
	{
		int jobSize = nextJobSize;
		listSize -= jobSize;
		nextJobSize = min(MAX_JOB_SIZE, listSize);

		// Copy the lists A and B to their respective memory buffers
		cl_int ret = clEnqueueWriteBuffer(cxt->command_queue, cxt->mi_mem_obj, CL_NON_BLOCKING, 0,
								   jobSize * sizeof(cl_uint), cxt->MI, 0, NULL, &complete_events[0]);
		DPRINTF("ret at %d is %d\n", __LINE__, ret);

		ret = clEnqueueWriteBuffer(cxt->command_queue, cxt->r_mem_obj, CL_NON_BLOCKING, 0,
								   jobSize * N_Size * sizeof(cl_uint), cxt->R, 0, NULL, &complete_events[1]);
		DPRINTF("ret at %d is %d\n", __LINE__, ret);

		ret = clEnqueueWriteBuffer(cxt->command_queue, cxt->m_mem_obj, CL_NON_BLOCKING, 0,
										  jobSize * N_Size * sizeof(cl_uint), M, 0, NULL, NULL);
		DPRINTF("ret at %d is %d\n", __LINE__, ret);

		if (!firstJob) clWaitForEvents(1, &complete_events[2]);
		else firstJob = 0;

		DPRINTF("before execution\n");
		// Execute the OpenCL kernel on the list
		size_t global_item_size = jobSize; // Process the entire lists
		ret = clEnqueueNDRangeKernel(cxt->command_queue, cxt->kernel, 1, NULL,
									 &global_item_size, NULL, 0, NULL, NULL);
		DPRINTF("ret at %d is %d\n", __LINE__, ret);

		if (nextJobSize > 0)
		{
			clWaitForEvents(2, complete_events);
			M += jobSize*N_Size;
			setup_fermat(N_Size, nextJobSize, M, cxt->MI, cxt->R);
		}

		DPRINTF("after execution\n");
		ret = clEnqueueReadBuffer(cxt->command_queue, cxt->is_prime_mem_obj, CL_NON_BLOCKING, 0,
								  jobSize * sizeof(cl_uint), is_prime, 0, NULL, &complete_events[2]);
		DPRINTF("ret at %d is %d\n", __LINE__, ret);
		DPRINTF("after copying\n");

		is_prime += jobSize;
	}
	clWaitForEvents(1, &complete_events[2]);

	//clFinish(cxt->command_queue);
}

void primeTestTerm(PrimeTestCxt* cxt)
{
	clFlush(cxt->command_queue);
	clFinish(cxt->command_queue);
	clReleaseProgram(cxt->program);
	clReleaseMemObject(cxt->m_mem_obj);
	clReleaseMemObject(cxt->mi_mem_obj);
	clReleaseMemObject(cxt->r_mem_obj);
	clReleaseMemObject(cxt->is_prime_mem_obj);
	clReleaseCommandQueue(cxt->command_queue);
	clReleaseContext(cxt->context);
	free(cxt->R);
	free(cxt->MI);
}
