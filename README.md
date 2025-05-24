# Oblicznie hashy MD5 na OpenMP i CUDA
## Projekt programowanie równoległe

##  Zależności
- cmake 3.27
- g++-13
- cuda
- openmp

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

Po pomyślnej kompilacji w katalogu `build` pojawią się 2 pliki wykonywalne:
- `main` z kodem aplikacji
- `main_test` z testami aplikacji