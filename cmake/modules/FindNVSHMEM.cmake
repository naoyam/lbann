# Exports the following variables
#
#   NVSHMEM_FOUND
#   NVSHMEM_INCLUDE_PATH
#   NVSHMEM_LIBRARIES
#
# Exports the following IMPORTED target
#
#   cuda::nvshmem
#

message(STATUS "NVSHMEM_DIRxx: ${NVSHMEM_DIR}")

find_path(NVSHMEM_INCLUDE_PATH nvshmem.h
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR} ${nvshmem_DIR} $ENV{nvshmem_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of NVSHMEM header."
  )
find_path(NVSHMEM_INCLUDE_PATH nvshmem.h)

message(STATUS "NVSHMEM_INCLUDE_PATH: ${NVSHMEM_INCLUDE_PATH}")

find_library(NVSHMEM_LIBRARY nvshmem
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR} ${nvshmem_DIR} $ENV{nvshmem_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The NVSHMEM library."
  )
find_library(NVSHMEM_LIBRARY nvshmem)

message(STATUS "NVSHMEM_LIBRARY: ${NVSHMEM_LIBRARY}")

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
  DEFAULT_MSG NVSHMEM_LIBRARY NVSHMEM_INCLUDE_PATH)

if (NOT TARGET cuda::nvshmem)

  add_library(cuda::nvshmem INTERFACE IMPORTED)

  set_property(TARGET cuda::nvshmem PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_PATH}")

  set_property(TARGET cuda::nvshmem PROPERTY
    INTERFACE_LINK_LIBRARIES "${NVSHMEM_LIBRARY}")

endif (NOT TARGET cuda::nvshmem)
