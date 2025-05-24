cmake_minimum_required(VERSION 3.15) # Or a more recent version

project(MyCudaOmpProject LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find VCPKG toolchain (assuming VCPKG_ROOT environment variable is set)
find_package(VCPKG REQUIRED)
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "VCPKG toolchain file")

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found with version: ${CUDA_VERSION}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_XX") # Replace XX with your target GPU architecture(s)
    enable_language(CUDA)
else()
    message(WARNING "CUDA not found. CUDA-specific code will not be built.")
endif()

# Find OpenMP (Modern CMake way)
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. OpenMP-specific code will not be built.")
endif()

# Find GTest
find_package(GTest REQUIRED)
if(GTEST_FOUND)
    message(STATUS "GTest found")
    include_directories(${GTEST_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "GTest not found. Please install it via vcpkg.")
endif()

# Define executable target
add_executable(my_app src/main.cpp src/some_code.cpp src/some_code.h)
target_link_libraries(my_app ${OpenMP_LIBRARIES})
if(CUDA_FOUND)
    target_link_libraries(my_app ${CUDA_LIBRARIES})
    cuda_add_library(cuda_lib SHARED src/some_code.cu) # Example CUDA library
    target_link_libraries(my_app cuda_lib)
endif()

# Define test executable target
add_executable(my_tests test/some_code_test.cpp src/some_code.cpp src/some_code.h)
target_link_libraries(my_tests GTest::gtest_main ${OpenMP_LIBRARIES})
if(CUDA_FOUND)
    target_link_libraries(my_tests ${CUDA_LIBRARIES})
    target_link_libraries(my_tests cuda_lib)
    cuda_add_library(cuda_test_lib SHARED test/some_cuda_test.cu) # Optional CUDA tests
    target_link_libraries(my_tests cuda_test_lib)
endif()

# Enable CTest integration
enable_testing()
add_test(NAME unit_tests COMMAND my_tests)