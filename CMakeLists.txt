cmake_minimum_required(VERSION 3.27) # Or a more recent version

set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)
project(MyCudaOmpProject LANGUAGES CUDA CXX)

# Find CUDA
# find_package(CUDA REQUIRED)
# if(CUDA_FOUND)
#     message(STATUS "CUDA found with version: ${CUDA_VERSION}")
#     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ")
#     enable_language(CUDA)
# else()
#     message(WARNING "CUDA not found. CUDA-specific code will not be built.")
# endif()

# Find OpenMP (Modern CMake way)
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. OpenMP-specific code will not be built.")
endif()

# Find GTest
include(GoogleTest)
# find_package(GTest REQUIRED)
# if(GTEST_FOUND)
#     message(STATUS "GTest found")
#     include_directories(${GTEST_INCLUDE_DIRS})
# else()
#     message(FATAL_ERROR "GTest not found. Please install it via vcpkg.")
# endif()

# Define executable target
add_executable(projekt_rownolegle src/main.cpp src/some_code.cpp src/some_code.h)
target_link_libraries(projekt_rownolegle ${OpenMP_LIBRARIES})
if(CUDA_FOUND)
    target_link_libraries(projekt_rownolegle ${CUDA_LIBRARIES})
    cuda_add_library(cuda_lib SHARED src/some_code.cu) # Example CUDA library
    target_link_libraries(projekt_rownolegle cuda_lib)
endif()

# Define test executable target
add_executable(projekt_rownolegle_testy test/some_code_test.cpp src/some_code.cpp src/some_code.h)
target_link_libraries(projekt_rownolegle_testy GoogleTest ${OpenMP_LIBRARIES})
if(CUDA_FOUND)
    target_link_libraries(projekt_rownolegle_testy ${CUDA_LIBRARIES})
    target_link_libraries(projekt_rownolegle_testy cuda_lib)
    cuda_add_library(cuda_test_lib SHARED test/some_cuda_test.cu) # Optional CUDA tests
    target_link_libraries(projekt_rownolegle_testy cuda_test_lib)
endif()

# Enable CTest integration
enable_testing()
add_test(NAME unit_tests COMMAND projekt_rownolegle_testy)