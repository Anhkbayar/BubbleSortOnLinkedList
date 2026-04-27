#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>

#include "../linked_list.h"
#include "../results.h"

// Array to list conversion
int *listToArray(Node *head, int n)
{
    int *arr = new int[n];
    Node *p = head;
    for (int i = 0; i < n; i++, p = p->next)
    {
        arr[i] = p->data;
    }
    return arr;
}

void arrayToList(Node *head, const int *arr, int n)
{
    Node *p = head;
    for (int i = 0; i < n; i++, p = p->next)
    {
        p->data = arr[i];
    }
}

// oddEvenSort
__global__ void oddEvenKernel(int *arr, int n, int phase)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int start = phase;
    int idx = start + 2 * i;
    if (idx + 1 < n && arr[idx] > arr[idx + 1])
    {
        int temp = arr[idx];
        arr[idx] = arr[idx + 1];
        arr[idx + 1] = temp;
    }
}

Node *oddEvenSort(
    Node *head,
    int threads,
    int blocks,
    double &computationTimeMs,
    double &executionTimeMs,
    double &dataTransferTimeMs)
{
    int n = listLength(head);
    if (n <= 1)
        return head;

    auto execStart = std::chrono::high_resolution_clock::now();
    int *h_arr = listToArray(head, n);

    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));

    // transfer time 1
    auto transferStart = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    auto transferEnd = std::chrono::high_resolution_clock::now();

    // computation time
    auto computationStart = std::chrono::high_resolution_clock::now();
    for (int pass = 0; pass < n; ++pass)
    {
        oddEvenKernel<<<blocks, threads>>>(d_arr, n, pass % 2);
    }

    cudaDeviceSynchronize();
    auto computationEnd = std::chrono::high_resolution_clock::now();

    auto transferStart2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    auto transferEnd2 = std::chrono::high_resolution_clock::now();

    cudaFree(d_arr);

    arrayToList(head, h_arr, n);

    delete[] h_arr;
    auto execEnd = std::chrono::high_resolution_clock::now();

    double hostToDeviceMs = std::chrono::duration<double, std::milli>(transferEnd - transferStart).count();
    double deviceToHostMs = std::chrono::duration<double, std::milli>(transferEnd2 - transferStart2).count();

    dataTransferTimeMs = hostToDeviceMs + deviceToHostMs;

    computationTimeMs = std::chrono::duration<double, std::milli>(computationEnd - computationStart).count();
    executionTimeMs = std::chrono::duration<double, std::milli>(execEnd - execStart).count();

    return head;
}

// benchmark
CudaResult benchmark(const std::string &name, int n)
{
    Node *list = createList(n);

    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;

    double computationTimeMs = 0.0;
    double executionTimeMs = 0.0;
    double dataTransferTimeMs = 0.0;

    auto t_start = std::chrono::high_resolution_clock::now();
    Node *sorted = oddEvenSort(
        list, 
        threads, 
        blocks, 
        computationTimeMs, 
        executionTimeMs, 
        dataTransferTimeMs);
        
    auto t_end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bool ok = isSorted(sorted, n);

    double throughput = 0.0;
    if (ms > 0)
    {
        throughput = n / (ms / 1000.0);
    }

    CudaResult result{
        name,
        n,
        blocks,
        threads,
        computationTimeMs,
        executionTimeMs,
        dataTransferTimeMs,
        throughput,
        ok};

    // std::cout << "First values: ";
    // printList(sorted);
    deleteList(sorted);
    return result;
}

// Main
int main()
{
    cudaFree(nullptr);

    std::cout << "--- CUDA Linked list Sort ---\n\n";

    constexpr int N_SMALL = 10000;
    constexpr int N_MEDIUM = 100000;
    constexpr int N_LARGE = 200000;

    printCudaResultHeader();
    for (int n : {N_SMALL, N_MEDIUM, N_LARGE})
    {
        CudaResult result = benchmark("Cuda Odd-Even", n);
        printCudaResult(result);
    }

    return 0;
}