cmake_minimum_required(VERSION 2.8)

#######################################
project(pydarknet)
set(CMAKE_BUILD_TYPE "Release")
#set(APPLE On)
option(CUDA "build with cuda" Off)

set(CFLAGS -Wall -O2)

if (APPLE)
    # MacPorts
    message("USING MACPORTS")
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )
add_definitions(-DOPENCV)
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "Checking Threads")
find_package( Threads REQUIRED )

if (CUDA)
    message(STATUS "Checking CUDA")
    find_package( CUDA REQUIRED )
    add_definitions(-DGPU)
    message("CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_NVCC_FLAGS --gpu-architecture=compute_20 --gpu-code=compute_20 --compiler-options ${CFLAGS})
endif()

#######################################

file(GLOB SOURCE_FILES
    "src/*.c"
    "src/*.h"
)

file(GLOB CUDA_SOURCE_FILES
    "src/*.cu"
)

if (CUDA)
    cuda_add_library(pydarknet SHARED
        ${CUDA_SOURCE_FILES}
    	${SOURCE_FILES}
    )
else ()
    add_library(pydarknet SHARED
        ${SOURCE_FILES}
    )
endif ()

#######################################
if (CUDA)
    #link_directories(
    #    /usr/local/cuda/lib64
    #)

    message("CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
    message("CMAKE_THREAD_LIBS_INIT = ${CMAKE_THREAD_LIBS_INIT}")
    message("CUDA_CUBLAS_LIBRARIES = ${CUDA_CUBLAS_LIBRARIES}")
    message("CUDA_curand_LIBRARY = ${CUDA_curand_LIBRARY}")
    

    target_link_libraries(pydarknet
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_curand_LIBRARY}
    )
else ()
    target_link_libraries(pydarknet
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif ()

