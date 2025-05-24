cmake_minimum_required(VERSION 3.27) # Or a more recent version

set(CMAKE_CXX_COMPILER g++-13)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_DIR "/usr/local/cuda-12.5/lib64/cmake")
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)
project(MyCudaOmpProject LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP COMPONENTS CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Define executable target
add_executable(projekt_rownolegle src/main.cpp src/some_code.cpp src/some_code.h src/some_code.cu)
target_link_libraries(projekt_rownolegle ${OpenMP_LIBRARIES})
target_link_libraries(projekt_rownolegle CUDA::cudart)

# Define test executable target
# include(GoogleTest)
# add_executable(projekt_rownolegle_testy test/some_code_test.cpp test/some_cuda_test.cu src/some_code.cpp src/some_code.h src/some_code.cu)
# target_link_libraries(projekt_rownolegle ${OpenMP_LIBRARIES})
# target_link_libraries(projekt_rownolegle_testy ${CUDA_LIBRARIES})

# Enable CTest integration
enable_testing()
add_test(NAME unit_tests COMMAND projekt_rownolegle_testy)