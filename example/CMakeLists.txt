add_executable(myexe myexe.cpp)

find_package(GFLAGS REQUIRED)
find_package(Eigen3 REQUIRED)

target_link_libraries (myexe PRIVATE SuiteSparse::CHOLMOD gflags Eigen3::Eigen)
target_compile_definitions(myexe PUBLIC GPU_BLAS DLONG)