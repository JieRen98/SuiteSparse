#include "cs.h"
int main (void)
{
    cs_ci *T, *A, *Eye, *AT, *C, *D ;
    int i, m ;
    T = cs_ci_load (stdin) ;               /* load triplet matrix T from stdin */
    printf ("T:\n") ; cs_ci_print (T, 0) ; /* print T */
    A = cs_ci_compress (T) ;               /* A = compressed-column form of T */
    printf ("A:\n") ; cs_ci_print (A, 0) ; /* print A */
    cs_ci_spfree (T) ;                     /* clear T */
    AT = cs_ci_transpose (A, 1) ;          /* AT = A' */
    printf ("AT:\n") ; cs_ci_print (AT, 0) ; /* print AT */
    m = A ? A->m : 0 ;                  /* m = # of rows of A */
    T = cs_ci_spalloc (m, m, m, 1, 1) ;    /* create triplet identity matrix */
    for (i = 0 ; i < m ; i++) cs_ci_entry (T, i, i, CS_COMPLEX_MAKE(1., 0.)) ;
    Eye = cs_ci_compress (T) ;             /* Eye = speye (m) */
    cs_ci_spfree (T) ;
    C = cs_ci_multiply (A, AT) ;           /* C = A*A' */
    D = cs_ci_add (C, Eye, CS_COMPLEX_MAKE(1., 0.), CS_COMPLEX_MAKE(cs_ci_norm (C), 0.)) ;   /* D = C + Eye*norm (C,1) */
    printf ("D:\n") ; cs_ci_print (D, 0) ; /* print D */
    cs_ci_spfree (A) ;                     /* clear A AT C D Eye */
    cs_ci_spfree (AT) ;
    cs_ci_spfree (C) ;
    cs_ci_spfree (D) ;
    cs_ci_spfree (Eye) ;
    return (0) ;
}
