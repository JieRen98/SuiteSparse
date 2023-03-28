#include <gflags/gflags.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>
#include <chrono>

#include "SuiteSparse_config.h"
#include "cholmod.h"
#include "cholmod_matrixops.h"

void run(const std::string &dataPath, int useGPU) {
    const char *matFile = dataPath.data();
    FILE *fp = fopen(matFile, "r");
    assert(fp != NULL);

    cholmod_sparse *A;
    cholmod_dense *x, *b;
    auto c = (cholmod_common *) malloc(sizeof(cholmod_common));

    cholmod_l_start(c); /* start CHOLMOD */
    c->supernodal = CHOLMOD_SUPERNODAL;
    c->method[0].ordering = CHOLMOD_METIS;
    c->postorder = 1;
    c->nmethods = 1;
    c->useGPU = useGPU;
    A = cholmod_l_read_sparse(fp, c); /* read in a matrix */
    fclose(fp);

    Eigen::Map<Eigen::SparseMatrix<double, Eigen::ColMajor, long>>
            _mapA_double{(int) A->nrow,
                         (int) A->ncol,
                         (int) A->nzmax,
                         (long *) A->p,
                         (long *) A->i,
                         (double *) A->x};

    Eigen::SparseMatrix<double, Eigen::ColMajor, long> mapA_double =
            _mapA_double.selfadjointView<Eigen::Upper>();
    mapA_double.makeCompressed();

    A->p = mapA_double.outerIndexPtr();
    A->i = mapA_double.innerIndexPtr();
    A->x = mapA_double.valuePtr();
    A->nzmax = mapA_double.nonZeros();

    cholmod_l_print_sparse(A, "A", c); /* print the matrix */

    if (A == NULL || A->stype == 0) /* A must be symmetric */
    {
        cholmod_l_free_sparse(&A, c);
        cholmod_l_finish(c);
        return;
    }

    x = cholmod_l_ones(A->nrow, 1, A->xtype,
                     c); /* b = ones(n,1) */
    b = cholmod_l_ones(A->nrow, 1, A->xtype,
                     c); /* b = ones(n,1) */
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> mapB{
            (double *) b->x, (long) b->nrow};
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> mapX{
            (double *) x->x, (long) x->nrow};

    mapB = mapA_double * mapX;

    size_t memory_inuse = c->memory_inuse;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto f = cholmod_l_analyze(A, c);
    cholmod_l_factorize(A, f, c);
    cholmod_dense *Y{nullptr}, *E{nullptr};
    cholmod_l_solve2(CHOLMOD_A, f, b, nullptr, &x, nullptr, &Y, &E, c);
    auto duration = std::chrono::system_clock::now() - start;
    cholmod_l_free_dense(&Y, c);
    cholmod_l_free_dense(&E, c);
    cholmod_l_gpu_stats(c);
    printf("Memory: %zu MB\n", (c->memory_inuse - memory_inuse) / 1024 / 1024);

    Eigen::Matrix<double, Eigen::Dynamic, 1> computeB = mapA_double * mapX;

    double residual = (mapB - computeB).norm();

    printf("Residual: %.4e, elapsed %ld ms\n", residual,
           std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

    A->p = _mapA_double.outerIndexPtr();
    A->i = _mapA_double.innerIndexPtr();
    A->x = _mapA_double.valuePtr();
    A->nzmax = _mapA_double.nonZeros();
    cholmod_l_free_sparse(&A, c);
    cholmod_l_free_dense(&x, c);
    cholmod_l_free_dense(&b, c);
    cholmod_l_finish(c); /* finish CHOLMOD */
    free(c);
}

DEFINE_string(dataPath, "/media/jie/Data/Dataset/Matrix/nd6k/nd6k.mtx", "Refinement Method");
DEFINE_int32(useGPU, 0, "Use GPU or Not");

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    run(FLAGS_dataPath, FLAGS_useGPU);
    return 0;
}
