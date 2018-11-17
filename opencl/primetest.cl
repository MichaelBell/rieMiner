void squareSimple(uint* P, const uint* R, int SIZE)
{
	const uint highbit = ((uint)1) << 31;

	uint T[(N_Size - 1) * 2];

	{
		uint cy = 0;
		for (int i = 0; i < SIZE - 1; ++i)
		{
			ulong p = (ulong)R[i + 1] * (ulong)R[0] + cy;
			T[i] = (uint)p;
			cy = (uint)(p >> 32);
		}
		T[SIZE - 1] = cy;
	}

	for (int j = 2; j < SIZE; ++j)
	{
		uint cy = 0;
		for (int i = j; i < SIZE; ++i)
		{
			ulong p = (ulong)R[i] * (ulong)R[j - 1];
			p += cy;
			p += T[i + j - 2];
			T[i + j - 2] = (uint)p;
			cy = (uint)(p >> 32);
		}
		T[SIZE + j - 2] = cy;
	}

	/* Better not to include this into the next loop as doing it first */
	/* avoids latency stalls. */
	for (int i = 0; i < SIZE; ++i)
	{
		ulong p = (ulong)R[i] * (ulong)R[i];
		P[2 * i] = (uint)p;
		P[2 * i + 1] = (uint)(p >> 32);
	}

	uint cy = 0;
	for (int i = 0; i < SIZE - 1; ++i)
	{
		uint t = T[2 * i] & highbit;
		ulong a = (ulong)P[2 * i + 1] + cy;
		a += T[2 * i] << 1;
		P[2 * i + 1] = (uint)a;
		cy = (t >> 31) + (uint)(a >> 32);

		t = T[2 * i + 1] & highbit;
		a = (ulong)P[2 * i + 2] + cy;
		a += T[2 * i + 1] << 1;
		P[2 * i + 2] = (uint)a;
		cy = (t >> 31) + (uint)(a >> 32);
	}
	P[2 * SIZE - 1] += cy;
}

uint bigAdd(uint* R, const uint* A, const uint* B, int SIZE)
{
	uint cy = 0;
	for (int i = 0; i < SIZE; ++i)
	{
		ulong a = (ulong)A[i] + (ulong)B[i];
		a += cy;
		R[i] = (uint)a;
		cy = (uint)(a >> 32);
	}
	return cy;
}

int bigSub(uint* R, const uint* A, const uint* B, int SIZE)
{
	int cy = 0;
	for (int i = 0; i < SIZE; ++i)
	{
		long a = (long)A[i] - (long)B[i];
		a += cy;
		R[i] = (uint)a;
		cy = (int)(a >> 32);
	}
	return cy;
}

bool lessThan(const uint* A, const uint* B, int SIZE)
{
	for (int i = SIZE - 1; i >= 0; --i)
	{
		if (A[i] < B[i]) return true;
		if (A[i] > B[i]) return false;
	}
	return false;
}

void toom2SquareHalf(uint* P, const uint* R, int SIZE)
{
	uint s = SIZE >> 1;
	uint n = SIZE - s;  // n==s or n==s+1

	const uint* a0 = R;
	const uint* a1 = R + n;
	uint* asm1 = P;

	if (s == n)
	{
		if (lessThan(a0, a1, n))
			bigSub(asm1, a1, a0, n);
		else
			bigSub(asm1, a0, a1, n);
	}
	else
	{
		if (a0[s] == 0 && lessThan(a0, a1, s))
		{
			bigSub(asm1, a1, a0, s);
			asm1[s] = 0;
		}
		else
		{
			asm1[s] = a0[s] + bigSub(asm1, a0, a1, s);
		}
	}

	uint* v0 = P;
	uint vm1[(N_Size >> 1) + 2];
	uint* vinf = P + 2 * n;

	squareSimple(vm1, asm1, n);
	squareSimple(vinf, a1, s);
	squareSimple(v0, a0, n);

	int cy  = bigAdd(P + 2*n, v0 + n, vinf, n);
	uint cy2 = cy + bigAdd(P + n, P + 2*n, v0, n);
	uint cy3 = bigAdd(P + 2*n, P + 2*n, vinf + n, s + s - n);
	for (int i = 2*n+s+s-n; i < 3*n && cy3 > 0; ++i)
	{
		ulong a = (ulong)P[i] + cy3;
		P[i] = (uint)a;
		cy3 = (uint)(a >> 32);
	}
	cy += cy3;

	cy += bigSub(P + n, P + n, vm1, 2*n);

	for (int i = 2*n; cy2 > 0; ++i)
	{
		ulong a = (ulong)P[i] + cy2;
		P[i] = (uint)a;
		cy2 = (uint)(a >> 32);
	}
	for (int i = 3*n; cy != 0; ++i)
	{
		long a = (long)P[i] + cy;
		P[i] = (uint)a;
		cy = (int)(a >> 32);
	}
}

void toom2SquareFull(uint* P, const uint* R)
{
	uint s = N_Size >> 1;
	uint n = N_Size - s;  // n==s or n==s+1

	const uint* a0 = R;
	const uint* a1 = R + n;
	uint* asm1 = P;

	if (s == n)
	{
		if (lessThan(a0, a1, n))
			bigSub(asm1, a1, a0, n);
		else
			bigSub(asm1, a0, a1, n);
	}
	else
	{
		if (a0[s] == 0 && lessThan(a0, a1, s))
		{
			bigSub(asm1, a1, a0, s);
			asm1[s] = 0;
		}
		else
		{
			asm1[s] = a0[s] + bigSub(asm1, a0, a1, s);
		}
		P[4*n-2] = 0;
		P[4*n-1] = 0;
	}

	uint* v0 = P;
	uint vm1[N_Size + 1];
	uint* vinf = P + 2 * n;

	//squareSimple(vm1, asm1, n);
	//squareSimple(vinf, a1, s);
	//squareSimple(v0, a0, n);
	toom2SquareHalf(vm1, asm1, n);
	toom2SquareHalf(vinf, a1, s);
	toom2SquareHalf(v0, a0, n);

	int cy  = bigAdd(P + 2*n, v0 + n, vinf, n);
	uint cy2 = cy + bigAdd(P + n, P + 2*n, v0, n);
	cy += bigAdd(P + 2*n, P + 2*n, vinf + n, n);
	cy += bigSub(P + n, P + n, vm1, 2*n);

	for (int i = 2*n; cy2 > 0; ++i)
	{
		ulong a = (ulong)P[i] + cy2;
		P[i] = (uint)a;
		cy2 = (uint)(a >> 32);
	}
	for (int i = 3*n; cy != 0; ++i)
	{
		long a = (long)P[i] + cy;
		P[i] = (uint)a;
		cy = (int)(a >> 32);
	}
}

__kernel void fermat_test(__global const uint *M_in, __global const uint *Mi_in, __global const uint *R_in, __global uint *is_prime) {

	uint R[N_Size];
	uint M[N_Size];

	// Get the index of the current element to be processed
	const int offset = get_global_id(0) * N_Size;

	for (int i = 0; i < N_Size; ++i)
	{
		M[i] = M_in[offset + i];
		R[i] = R_in[offset + i];
	}

	const uint shift = clz(M[N_Size - 1]);
	const uint highbit = ((uint)1) << 31;
	uint startbit = highbit >> shift;

	const uint mi = Mi_in[get_global_id(0)];

	int en = N_Size;
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
				//squareSimple(P, R, N_Size);
				toom2SquareFull(P, R);

				//if (mpn_redc_1(rp, pp, mp, mn, mi) != 0)
				//  mpn_sub_n(rp, rp, mshifted, n);
				for (int j = 0; j < N_Size; ++j)
				{
					uint cy = 0;
					uint v = P[j] * mi;
					for (int i = 0; i < N_Size; ++i)
					{
						ulong p = (ulong)M[i] * (ulong)v + cy;
						p += P[i + j];
						P[i + j] = (uint)p;
						cy = (uint)(p >> 32);
					}
					R[j] = cy;
				}

				{
					uint cy = 0;
					for (int i = 0; i < N_Size; ++i)
					{
						ulong a = (ulong)R[i] + (ulong)P[i + N_Size];
						if (E & bit) a <<= 1;
						a += cy;
						R[i] = (uint)a;
						cy = (uint)(a >> 32);
					}

					while (cy != 0)
					{
						int borrow = 0;
						uint last_shifted = 0;
						for (int i = 0; i < N_Size; ++i)
						{
							long a = R[i];
							uint b;
							if (shift != 0)
							{
								b = (M[i] << shift) | last_shifted;
								last_shifted = M[i] >> (32 - shift);
							}
							else
							{
								b = M[i];
							}
							a = a - (long)b + borrow;
							R[i] = (uint)a;
							borrow = (int)(a >> 32);
						}
						cy += borrow;
					}
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
		for (int j = 0; j < N_Size; ++j)
		{
			uint cy = 0;
			uint v = T[j] * mi;
			for (int i = 0; i < N_Size; ++i)
			{
				ulong p = (ulong)M[i] * (ulong)v + cy;
				p += T[i + j];
				T[i + j] = (uint)p;
				cy = (uint)(p >> 32);
			}
			R[j] = cy;
		}

		{
			uint cy = 0;
			for (int i = 0; i < N_Size; ++i)
			{
				ulong a = (ulong)R[i] + cy;
				a += T[i + N_Size];
				R[i] = (uint)a;
				cy = (uint)(a >> 32);
			}

			if (cy != 0)
			{
				int borrow = 0;
				uint last_shifted = 0;
				for (int i = 0; i < N_Size; ++i)
				{
					long a = R[i];
					uint b;
					if (shift != 0)
					{
						b = (M[i] << shift) | last_shifted;
						last_shifted = M[i] >> (32 - shift);
					}
					else
					{
						b = M[i];
					}
					a = a - (long)b + borrow;
					R[i] = (uint)a;
					borrow = (int)(a >> 32);
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

	is_prime[get_global_id(0)] = result;
}
