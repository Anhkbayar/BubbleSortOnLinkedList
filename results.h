#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>


struct CpuResult{
    std::string type;
    int size;

    int threadCount;
    double timeMs;
    double throughput;
    double speedup;
    bool sortedOk;
};


struct CudaResult{
    std::string type;
    int size;

    int blocks;
    int threadsPerBlock;
    double timeMs;
    double throughput;
    bool sortedOk;
};

// Print table header for CPU results
inline void printCpuResultHeader() {
    std::cout
        << std::left
        << std::setw(15) << "Type"
        << std::setw(12) << "Size"
        << std::setw(15) << "Threads"
        << std::setw(15) << "Time(ms)"
        << std::setw(18) << "Throughput"
        << std::setw(12) << "Speedup"
        << std::setw(10) << "Sorted"
        << '\n';

    std::cout << std::string(97, '-') << '\n';
}

// Print one CPU result row
inline void printCpuResult(const CpuResult& result) {
    std::cout
        << std::left
        << std::setw(15) << result.type
        << std::setw(12) << result.size
        << std::setw(15) << result.threadCount
        << std::setw(15) << std::fixed << std::setprecision(3) << result.timeMs
        << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput
        << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup
        << std::setw(10) << (result.sortedOk ? "Yes" : "No")
        << '\n';
}

// Print table header for CUDA results
inline void printCudaResultHeader() {
    std::cout
        << std::left
        << std::setw(15) << "Type"
        << std::setw(12) << "Size"
        << std::setw(12) << "Blocks"
        << std::setw(20) << "Threads/Block"
        << std::setw(15) << "Time(ms)"
        << std::setw(18) << "Throughput"
        << std::setw(10) << "Sorted"
        << '\n';

    std::cout << std::string(102, '-') << '\n';
}

// Print one CUDA result row
inline void printCudaResult(const CudaResult& result) {
    std::cout
        << std::left
        << std::setw(15) << result.type
        << std::setw(12) << result.size
        << std::setw(12) << result.blocks
        << std::setw(20) << result.threadsPerBlock
        << std::setw(15) << std::fixed << std::setprecision(3) << result.timeMs
        << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput
        << std::setw(10) << (result.sortedOk ? "Yes" : "No")
        << '\n';
}