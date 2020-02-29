
# Parallel Tests
# add_executable( lbann-bin lbann.cpp )
# target_link_libraries(lbann-bin lbann )
# #target_link_libraries(lbann-bin cuda::toolkit)
# #target_link_libraries(lbann-bin MPI::MPI_CXX)
# set_target_properties(lbann-bin PROPERTIES OUTPUT_NAME lbann)

add_executable( test-with-lbann lbann_nvshmem.cu )
target_link_libraries(test-with-lbann lbann )
target_link_libraries(test-with-lbann cuda::toolkit)
target_link_libraries(test-with-lbann cuda::nvshmem)
target_link_libraries(test-with-lbann MPI::MPI_CXX)
set_property(TARGET test-with-lbann APPEND PROPERTY
  CUDA_SEPARABLE_COMPILATION ON)

add_executable( test-without-lbann lbann_nvshmem.cu )
target_link_libraries(test-without-lbann cuda::toolkit)
target_link_libraries(test-without-lbann cuda::nvshmem)
target_link_libraries(test-without-lbann MPI::MPI_CXX)
set_property(TARGET test-without-lbann APPEND PROPERTY
  CUDA_SEPARABLE_COMPILATION ON)
