# Oblicznie hashy MD5 na OpenMP i CUDA
## Projekt programowanie równoległe

##  Zależności
- cmake 3.27
- g++-13
- cuda
- openmp
- google benchmark
- gtest

Możliwe, że na innym systemie zależności będą w innym miejscu. Należy podmienić te linie w `CMakeLists.txt`
```
set(CMAKE_CXX_COMPILER g++-13) # required for cuda
set(CUDA_DIR "/usr/local/cuda-12.5/lib64/cmake")
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)
```

## Budowanie

1.  **Utworzenie katalogu budowania**
    ```bash
    mkdir build
    cd build
    ```

2.  **Konfiguracja z CMake**
    ```bash
    cmake .. -DCMAKE_BUILD_TYPE=Debug
    ```

3.  **Kompilacja**
    ```bash
    make -j$(nproc) # lub ninja
    ```

## Uruchomienie

Po pomyślnej kompilacji w katalogu `build` pojawią się 3 pliki wykonywalne:
- `main` z kodem aplikacji
- `main_test` z testami jednostkowymi funkcji
- `main_benchmark` z testami wydajności poszczególnych implementacji