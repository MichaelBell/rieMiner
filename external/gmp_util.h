/* Useful utilities, copied from GMP
   GMP code is used under the GNU GPLv2 license. */

#pragma once

#define add_ssaaaa(sh, sl, ah, al, bh, bl) \
  __asm__ ("adds\t%1, %x4, %5\n\tadc\t%0, %x2, %x3"                     \
		             : "=r" (sh), "=&r" (sl)                                      \
			                : "rZ" ((uint64_t)(ah)), "rZ" ((uint64_t)(bh)),                \
					             "%r" ((uint64_t)(al)), "rI" ((uint64_t)(bl)) __CLOBBER_CC)
#define sub_ddmmss(sh, sl, ah, al, bh, bl) \
	  __asm__ ("subs\t%1, %x4, %5\n\tsbc\t%0, %x2, %x3"                     \
			             : "=r,r" (sh), "=&r,&r" (sl)                                 \
				                : "rZ,rZ" ((uint64_t)(ah)), "rZ,rZ" ((uint64_t)(bh)),          \
						             "r,Z"   ((uint64_t)(al)), "rI,r"  ((uint64_t)(bl)) __CLOBBER_CC)
#define umul_ppmm(ph, pl, m0, m1) \
	  do {                                                                  \
		      uint64_t __m0 = (m0), __m1 = (m1);                                   \
		      __asm__ ("umulh\t%0, %1, %2" : "=r" (ph) : "r" (__m0), "r" (__m1)); \
		      (pl) = __m0 * __m1;                                                 \
		    } while (0)

/* Dividing (NH, NL) by D, returning the remainder only. Unlike
   udiv_qrnnd_preinv, works also for the case NH == D, where the
   quotient doesn't quite fit in a single limb. */
#define udiv_rnnd_preinv(r, nh, nl, d, di)                              \
  do {                                                                  \
    mp_limb_t _qh, _ql, _r, _mask;                                      \
    umul_ppmm (_qh, _ql, (nh), (di));                                   \
    if (__builtin_constant_p (nl) && (nl) == 0)                         \
      {                                                                 \
        _r = ~(_qh + (nh)) * (d);                                       \
        _mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */     \
        _r += _mask & (d);                                              \
      }                                                                 \
    else                                                                \
      {                                                                 \
        add_ssaaaa (_qh, _ql, _qh, _ql, (nh) + 1, (nl));                \
        _r = (nl) - _qh * (d);                                          \
        _mask = -(mp_limb_t) (_r > _ql); /* both > and >= are OK */     \
        _r += _mask & (d);                                              \
        if (__GMP_UNLIKELY (_r >= (d)))                                 \
          _r -= (d);                                                    \
      }                                                                 \
    (r) = _r;                                                           \
  } while (0)

