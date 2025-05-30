cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++-13) # required for cuda
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set cuda configuration
# set(CUDA_DIR "/usr/local/cuda-12.5/lib64/cmake")
set(CMAKE_CUDA_ARCHITECTURES 50)
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

project(ProjektRownolegle LANGUAGES CUDA CXX)

# Dependencies (all installed using the package manager)
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP COMPONENTS CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# include(GoogleTest REQUIRED)
# set(GTEST_LIBRARIES gtest gtest_main)

find_package(benchmark REQUIRED)

# Compilation flags
include ( CheckCXXCompilerFlag )
foreach ( FLAG -Wall -Wextra -Wno-missing-field-initializers -Wno-vla )
  check_cxx_compiler_flag ( "${FLAG}" COMPILER_SUPPORTS_${FLAG} )
  if ( COMPILER_SUPPORTS_${FLAG} )
    add_compile_options ( ${FLAG} )
  endif ( )
endforeach ( )

# Glob to list all source files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE PROJECT_TEST_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu" "test/*.cpp" "test/*.cu")
get_filename_component(full_path_main_cpp src/main.cpp ABSOLUTE)
list(REMOVE_ITEM PROJECT_TEST_SOURCES ${full_path_main_cpp})
file(GLOB_RECURSE PROJECT_BENCHMARK_SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu" "benchmark/*.cpp" "benchmark/*.cu")
list(REMOVE_ITEM PROJECT_BENCHMARK_SOURCES ${full_path_main_cpp})
message(${PROJECT_SOURCES})
message(${PROJECT_TEST_SOURCES})

# Define executable target
add_executable(main) 
target_sources(main PRIVATE ${PROJECT_SOURCES})
target_include_directories(main PRIVATE "src")
target_link_libraries(main ${OpenMP_LIBRARIES} CUDA::cudart)

# # Define test executable target
# add_executable(main_test)
# # target_sources(main_test PRIVATE ${PROJECT_SOURCES})
# target_sources(main_test PRIVATE ${PROJECT_TEST_SOURCES})
# target_include_directories(main_test PRIVATE "src")
# target_include_directories(main_test PRIVATE "test")
# target_link_libraries(main_test ${GTEST_LIBRARIES} ${OpenMP_LIBRARIES}  CUDA::cudart)
# enable_testing(false)
# gtest_discover_tests(main_test)

# Define benchmark executable target
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_BUILD_TYPE Release)
add_executable(main_benchmark)
# target_sources(main_test PRIVATE ${PROJECT_SOURCES})
target_sources(main_benchmark PRIVATE ${PROJECT_BENCHMARK_SOURCES})
target_include_directories(main_benchmark PRIVATE "src")
target_include_directories(main_benchmark PRIVATE "benchmark")
target_link_libraries(main_benchmark ${OpenMP_LIBRARIES}  CUDA::cudart benchmark::benchmark)