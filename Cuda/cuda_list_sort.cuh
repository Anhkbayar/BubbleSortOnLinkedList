#pragma once
// cuda_list_sort.cuh
// Strategy: linked list lives on the CPU.
// To sort on GPU we:
//   1. Flatten list  → host array
//   2. Copy          → device array
//   3. Sort          on GPU (Thrust or custom kernel)
//   4. Copy back     → host array
//   5. Write values  back into the list nodes

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include "linked_list.h"   // Node, createList, etc.

// ── GPU error checking macro ──────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ── Flatten list → raw array (CPU) ───────────────────────────────────────────
inline int* listToArray(Node* head, int n) {
    int* arr = new int[n];
    Node* p  = head;
    for (int i = 0; i < n; ++i, p = p->next)
        arr[i] = p->val;
    return arr;
}

// ── Write sorted array values back into list nodes (CPU) ─────────────────────
inline void arrayToList(Node* head, const int* arr, int n) {
    Node* p = head;
    for (int i = 0; i < n; ++i, p = p->next)
        p->val = arr[i];
}

// ═════════════════════════════════════════════════════════════════════════════
//  1.  Thrust sort  (easiest, recommended)
// ═════════════════════════════════════════════════════════════════════════════
inline Node* thrustSort(Node* head) {
    int n = listLength(head);
    if (n <= 1) return head;

    // flatten
    int* h_arr = listToArray(head, n);

    // host → device
    thrust::device_vector<int> d_vec(h_arr, h_arr + n);

    // parallel sort on GPU
    thrust::sort(d_vec.begin(), d_vec.end());

    // device → host
    thrust::copy(d_vec.begin(), d_vec.end(), h_arr);

    // write back into list
    arrayToList(head, h_arr, n);

    delete[] h_arr;
    return head;
}

// ═════════════════════════════════════════════════════════════════════════════
//  2.  Custom CUDA kernel — Odd-Even (Brick) Sort
//      Each thread handles one compare-and-swap pair.
//      Runs n passes alternating odd/even phases → O(n) passes, O(1) per thread.
// ═════════════════════════════════════════════════════════════════════════════
__global__ void oddEvenKernel(int* arr, int n, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // phase 0 = even pairs (0,1),(2,3),...
    // phase 1 = odd  pairs (1,2),(3,4),...
    int start = phase;          // first element of pair for this thread
    int idx   = start + 2 * i; // each thread owns one pair
    if (idx + 1 < n && arr[idx] > arr[idx + 1]) {
        int tmp      = arr[idx];
        arr[idx]     = arr[idx + 1];
        arr[idx + 1] = tmp;
    }
}

inline Node* cudaOddEvenSort(Node* head) {
    int n = listLength(head);
    if (n <= 1) return head;

    // flatten
    int* h_arr = listToArray(head, n);

    // host → device
    int* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (n / 2 + threads - 1) / threads;

    // n passes: alternating even/odd phases
    for (int pass = 0; pass < n; ++pass)
        oddEvenKernel<<<blocks, threads>>>(d_arr, n, pass % 2);

    CUDA_CHECK(cudaDeviceSynchronize());

    // device → host
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_arr));

    arrayToList(head, h_arr, n);

    delete[] h_arr;
    return head;
}

// ═════════════════════════════════════════════════════════════════════════════
//  3.  Bitonic Sort  — O(log²n) passes, each pass fully parallel
//      Requires n to be a power of two (pads if needed).
// ═════════════════════════════════════════════════════════════════════════════
__global__ void bitonicStepKernel(int* arr, int j, int k) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj  = i ^ j;
    if (ixj > i) {
        bool ascending = (i & k) == 0;
        if ((arr[i] > arr[ixj]) == ascending) {
            int tmp  = arr[i];
            arr[i]   = arr[ixj];
            arr[ixj] = tmp;
        }
    }
}

inline int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline Node* cudaBitonicSort(Node* head) {
    int n = listLength(head);
    if (n <= 1) return head;

    int np     = nextPow2(n);           // pad to power of two
    int* h_arr = new int[np];
    Node* p    = head;
    for (int i = 0; i < n;  ++i, p = p->next) h_arr[i] = p->val;
    for (int i = n; i < np; ++i) h_arr[i] = INT_MAX;   // padding

    int* d_arr;
    CUDA_CHECK(cudaMalloc(&d_arr, np * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_arr, h_arr, np * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (np + threads - 1) / threads;

    for (int k = 2; k <= np; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            bitonicStepKernel<<<blocks, threads>>>(d_arr, j, k);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, np * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_arr));

    arrayToList(head, h_arr, n);   // only write back original n values

    delete[] h_arr;
    return head;
}
