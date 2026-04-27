# Дан холбоост жагсаалт дээр бөмбөлгөн эрэмбэлэлт хийх

Энэхүү төсөл дээр эдгээр аргуудыг ашиглан бөмбөлгөн жагсаалтын төрөл болох тэгш-сондгой эрэмблэлтыг дан холбоост жагсаалт дээр хэрэгжүүлэв:

- Sequential CPU
- Multithreaded CPU болон OpenMP
- CUDA GPU 

## Шаардлагатай сангууд

Компайл хийхээс өмнө, эдгээр зүйлс суулгагдсан байх шаардлагатай:

- C++ компайлер
    - Window: MingGW / MSYS2
    - Linux: `g++`
- OpenMP дэмжлэг
- NVIDIA CUDA Toolkit for the CUDA version
- Git

Cuda татагдсан эсэхийг шалгахдаа:
```bash
nvcc --version
```

C++ компайлер татагдсан эсэхийг шалгахдаа:
```bash
g++ --version
```

## Ажиллуулах заавар
Sequential CPU болон OpenMP
```bash
cd OPENMP
g++ -fopenmp main.cpp -o bubbleSortOpenMP
./bubbleSortOpenMP
```

Multithread CPU
```bash
cd Multithread
g++ main.cpp -o bubbleSortMultiThread
./bubbleSortMultiThread
```

Cuda GPU
```bash
cd Cuda
nvcc -arch=sm_89 -o bubbleSortCuda LinkedListCuda.cu
./bubbleSortCuda
```

sm_89 нь RTX4050 дээрх compute capability-д зориулж компайл хийлгэж байгаа тул өөрийн GPU-ний compute capability-г
```bash
nvidia-smi --query-gpu=compute_cap
```
гэж харна уу.
