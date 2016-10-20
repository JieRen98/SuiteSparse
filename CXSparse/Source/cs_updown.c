#include "cs.h"
/* sparse Cholesky update/downdate, L*L' + sigma*w*w' (sigma = +1 or -1) */
CS_INT cs_updown (cs *L, CS_INT sigma, const cs *C, const CS_INT *parent)
{
    CS_INT n, p, f, j, *Lp, *Li, *Cp, *Ci ;
    CS_ENTRY *Lx, *Cx, alpha, gamma, w1, w2, *w ;
    double beta = 1, beta2 = 1, delta ;
#ifdef CS_COMPLEX
    cs_complex_t phase ;
#endif
    if (!CS_CSC (L) || !CS_CSC (C) || !parent) return (0) ;  /* check inputs */
    Lp = L->p ; Li = L->i ; Lx = L->x ; n = L->n ;
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    if ((p = Cp [0]) >= Cp [1]) return (1) ;        /* return if C empty */
    w = (CS_ENTRY *)cs_malloc (n, sizeof (CS_ENTRY)) ;          /* get workspace */
    if (!w) return (0) ;                            /* out of memory */
    f = Ci [p] ;
    for ( ; p < Cp [1] ; p++) f = CS_MIN (f, Ci [p]) ;  /* f = min (find (C)) */
    for (j = f ; j != -1 ; j = parent [j]) w [j] = CS_ZERO() ;  /* clear workspace w */
    for (p = Cp [0] ; p < Cp [1] ; p++) w [Ci [p]] = Cx [p] ; /* w = C */
    for (j = f ; j != -1 ; j = parent [j])          /* walk path f up to root */
    {
        p = Lp [j] ;
        alpha = CS_DIV(w [j], Lx [p]) ;                    /* alpha = w(j) / L(j,j) */
        beta2 = beta*beta + (sigma)*CS_REAL(CS_MUL(alpha, CS_CONJ(alpha))) ;
        if (beta2 <= 0) break ;                     /* not positive definite */
        beta2 = sqrt (beta2) ;
        delta = (sigma > 0) ? (beta / beta2) : (beta2 / beta) ;
        gamma = CS_DIV(CS_MUL(CS_MAKE_ENTRY((double)sigma), CS_CONJ(alpha)), CS_MAKE_ENTRY(beta2 * beta)) ;
        Lx [p] = CS_ADD(CS_MUL(CS_MAKE_ENTRY(delta), Lx [p]), ((sigma > 0) ? (CS_MUL(gamma, w [j])) : CS_ZERO())) ;
        beta = beta2 ;
#ifdef CS_COMPLEX
        phase = CS_DIV(CS_MAKE_ENTRY(CS_ABS (Lx [p])), Lx [p]) ;  /* phase = abs(L(j,j))/L(j,j)*/
        Lx [p] = CS_MUL(Lx[p], phase) ;                   /* L(j,j) = L(j,j) * phase */
#endif
        for (p++ ; p < Lp [j+1] ; p++)
        {
            w1 = w [Li [p]] ;
            w [Li [p]] = w2 = CS_SUB(w1, CS_MUL(alpha, Lx [p])) ;
            Lx [p] = CS_ADD(CS_MUL(CS_MAKE_ENTRY(delta), Lx [p]), CS_MUL(gamma, ((sigma > 0) ? w1 : w2))) ;
#ifdef CS_COMPLEX
            Lx [p] = CS_MUL(Lx[p], phase) ;               /* L(i,j) = L(i,j) * phase */
#endif
        }
    }
    cs_free (w) ;
    return (beta2 > 0) ;
}
