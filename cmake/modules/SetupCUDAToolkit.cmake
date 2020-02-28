# This handles the non-compiler aspect of the CUDA toolkit.
# NCCL and cuDNN are handled separately.

if (NOT CUDA_FOUND)
  find_package(CUDA REQUIRED)
endif ()

find_package(NVTX REQUIRED)
find_package(cuDNN REQUIRED)
find_package(NVSHMEM)

if (NOT TARGET cuda::toolkit)
  add_library(cuda::toolkit INTERFACE IMPORTED)
endif ()

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES cuda::cudnn cuda::nvtx)

if (NVSHMEM_FOUND)
  set_property(TARGET cuda::toolkit APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES cuda::nvshmem)
  #set_property(TARGET cuda::toolkit APPEND PROPERTY
  #INTERFACE_LINK_LIBRARIES ${CUDA_cudadevrt_LIBRARY})
  #set_property(TARGET cuda::toolkit APPEND PROPERTY
  #INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>)
  string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70")
endif ()
