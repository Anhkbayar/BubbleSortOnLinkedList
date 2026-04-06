// main_cuda.cu
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "../linked_list.h"      // createList, printList, freeList, listLength
#include "cuda_list_sort.cuh" // thrustSort, cudaOddEvenSort, cudaBitonicSort

using SortFn = Node* (*)(Node*);

void benchmark(const std::string& name, SortFn fn, int n) {
    Node* list = createList(n);

    auto t0     = std::chrono::high_resolution_clock::now();
    Node* sorted = fn(list);
    auto t1     = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::left  << std::setw(22) << name
              << "  n="     << std::setw(8)  << n
              << "  time="  << std::fixed << std::setprecision(3) << ms << " ms\n";

    std::cout << "  first values: ";
    printList(sorted, 8);

    freeList(sorted);
}

int main() {
    // Warm up GPU (first CUDA call has driver init overhead)
    cudaFree(nullptr);

    std::cout << "=== CUDA Linked-list Sort Demo ===\n\n";

    constexpr int N_SMALL  =    10'000;
    constexpr int N_MEDIUM =   100'000;
    constexpr int N_LARGE  = 1'000'000;

    for (int n : {N_SMALL, N_MEDIUM, N_LARGE}) {
        std::cout << "-- n=" << n << " --\n";
        benchmark("Thrust Sort",       thrustSort,       n);
        if (n <= N_MEDIUM)
            benchmark("Odd-Even Sort", cudaOddEvenSort,  n);  // too slow for 1M
        benchmark("Bitonic Sort",      cudaBitonicSort,  n);
        std::cout << "\n";
    }

    return 0;
}
