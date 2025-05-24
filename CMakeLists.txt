cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++-13) # required for cuda
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set cuda configuration
set(CUDA_DIR "/usr/local/cuda-12.5/lib64/cmake")
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

project(ProjektRownolegle LANGUAGES CUDA CXX)

# Dependencies (all installed using the package manager)
find_package(CUDAToolkit REQUIRED)

find_package(OpenMP COMPONENTS CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include(GoogleTest REQUIRED)
set(GTEST_LIBRARIES gtest gtest_main)

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
list(REMOVE_ITEM PROJECT_TEST_SOURCES "src/main.cpp")

# Define executable target
add_executable(main) 
target_sources(main PRIVATE ${PROJECT_SOURCES})
target_include_directories(main PRIVATE "src")
target_link_libraries(main ${OpenMP_LIBRARIES} CUDA::cudart)

# Define test executable target
add_executable(main_test)
# target_sources(main_test PRIVATE ${PROJECT_SOURCES})
target_sources(main_test PRIVATE ${PROJECT_TEST_SOURCES})
target_include_directories(main_test PRIVATE "src")
target_include_directories(main_test PRIVATE "test")
target_link_libraries(main_test ${GTEST_LIBRARIES} ${OpenMP_LIBRARIES}  CUDA::cudart)

# Enable CTest integration
enable_testing()
add_test(NAME unit_tests COMMAND main_test)