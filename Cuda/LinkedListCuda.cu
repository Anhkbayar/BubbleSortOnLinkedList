#include <chrono>
#include <iostream>
#include <string>
#include <iomanip>

#include "../linked_list.h"

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

Node *oddEvenSort(Node *head)
{
    int n = listLength(head);
    if (n <= 1)
        return head;

    int *h_arr = listToArray(head, n);

    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;

    for (int pass = 0; pass < n; ++pass)
    {
        oddEvenKernel<<<blocks, threads>>>(d_arr, n, pass % 2);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);

    arrayToList(head, h_arr, n);

    delete[] h_arr;
    return head;
}

// benchmark
using SortFn = Node *(*)(Node *);

void benchmark(const std::string &name, SortFn fn, int n)
{
    Node *list = createList(n);

    auto t_start = std::chrono::high_resolution_clock::now();
    Node *sorted = fn(list);
    auto t_end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bool ok = isSorted(sorted, n);

    std::cout << std::left << std::setw(22) << name
              << "n=" << std::setw(8) << n
              << "time= " << std::fixed << std::setprecision(3) << ms << "ms\n"
              << "  [" << (ok ? "OK" : "FAIL") << "]\n";

    // std::cout << "First values: ";
    // printList(sorted);
    deleteList(sorted);
}

// Main
int main()
{
    cudaFree(nullptr);

    std::cout << "--- CUDA Linked list Sort ---\n\n";

    constexpr int N_SMALL = 10000;
    constexpr int N_MEDIUM = 100000;
    constexpr int N_LARGE = 1000000;

    for (int n : {N_SMALL, N_MEDIUM, N_LARGE})
    {
        std::cout << "Ugugdliin too = " << n << "--\n";
        benchmark("Odd-Even sort", oddEvenSort, n);
    }

    return 0;
}