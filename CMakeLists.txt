cmake_minimum_required(VERSION 2.8)

#######################################
project(pydarknet)
set(CMAKE_BUILD_TYPE "Release")
set(APPLE On)
set(CUDA On)

if (APPLE)
    #MacPorts
    message("USING MACPORTS")
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# if (CUDA)
#     find_package( CUDA REQUIRED )
#     set(
#         CUDA_NVCC_FLAGS
#         ${CUDA_NVCC_FLAGS}
#         --gpu-architecture=compute_20 --gpu-code=compute_20
#     )
# endif()

#######################################

file(GLOB SOURCE_FILES
    "src/*.c"
)

# file(GLOB CUDA_SOURCE_FILES
#     "src/*.cu"
# )

add_library(pydarknet SHARED
    src/pydarknet.cpp
	${SOURCE_FILES}
)

# if (CUDA)
#     cuda_add_library(pydarknetcuda SHARED
#         src/pydarknet.cpp
#         ${CUDA_SOURCE_FILES}
#     )
# endif()

#######################################
target_link_libraries(pydarknet
    ${OpenCV_LIBS}
)
