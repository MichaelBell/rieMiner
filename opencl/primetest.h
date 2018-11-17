#include <stdint.h>

struct PrimeTestCxt* primeTestInit();
void primeTestTerm(struct PrimeTestCxt* cxt);

void primeTest(struct PrimeTestCxt* cxt, int N_Size, int LIST_SIZE, const uint32_t* M, uint32_t* is_prime, void (*workFn)(void*), void*);
