# My CUDA and OpenMP Project

This project provides a basic structure for a C++ application that utilizes CUDA for GPU acceleration and OpenMP for multi-core parallelism. It also includes unit testing using Google Test (gtest) and manages dependencies with vcpkg.

## Prerequisites

Before you begin, ensure you have the following installed:

* **CMake (>= 3.15):** For building the project.
* **A C++ Compiler with C++17 Support (e.g., GCC, Clang):** For compiling the C++ code.
* **CUDA Toolkit:** NVIDIA's parallel computing platform and programming model. Ensure the drivers are also installed.
* **OpenMP Support:** Usually integrated with modern C++ compilers.
* **vcpkg:** Microsoft's C++ Package Manager. Follow the installation instructions on the [vcpkg repository](https://github.com/microsoft/vcpkg).

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your_repository_url>
    cd my_cuda_omp_project
    ```

2.  **Initialize and update vcpkg submodules (if you added vcpkg as a submodule):**
    ```bash
    git submodule update --init --recursive
    ```

3.  **Set the `VCPKG_ROOT` environment variable:**
    ```bash
    export VCPKG_ROOT=/path/to/your/vcpkg # Replace with your actual vcpkg path
    ```
    You might want to add this to your shell's configuration file (e.g., `.bashrc`, `.zshrc`).

## Building the Project

1.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

2.  **Configure the project with CMake:**
    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Debug # Or Release for optimized builds
    ```

3.  **Build the project:**
    ```bash
    make -j$(nproc) # Compile in parallel using all available processors
    ```

## Running the Application

After a successful build, the executable `my_app` will be in the `build` directory (or a subdirectory within `build` depending on your CMake configuration).

```bash
./my_app
````

## Running the Tests

The unit tests executable `my_tests` will also be in the `build` directory.

```bash
./my_tests
```

## Project Structure

```
my_cuda_omp_project/
├── CMakeLists.txt         # CMake build configuration file
├── .vscode/               # Optional: VS Code settings
│   └── settings.json
├── src/                   # Source code files
│   ├── main.cpp
│   ├── some_code.cpp
│   └── some_code.h
├── test/                  # Unit test files
│   └── some_code_test.cpp
├── vcpkg.json             # vcpkg dependency manifest
└── vcpkg-configuration.json # Optional: vcpkg custom registry configuration
└── README.md              # This file
└── .gitignore             # Specifies intentionally untracked files that Git should ignore
```

## Adding Dependencies

To add more libraries to your project:

1.  **Find the package name in the vcpkg registry:** You can search on the [vcpkg website](https://www.google.com/search?q=https://vcpkg.io/packages) or using the `vcpkg search <package-name>` command.
2.  **Add the dependency to the `"dependencies"` array in `vcpkg.json`**.
3.  **Reconfigure CMake:** Run `cmake ..` in your `build` directory again.
4.  **Update your `CMakeLists.txt`** to `find_package()` the new library and `target_link_libraries()` to link against it.
5.  **Rebuild the project:** Run `make -j$(nproc)` in your `build` directory.

## Contributing

[Add your contribution guidelines here if applicable]

## License

[Add your project's license information here]
