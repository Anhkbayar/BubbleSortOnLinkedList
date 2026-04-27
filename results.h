#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>

struct CpuResult
{
    std::string type;
    int size;

    int threadCount;
    double computationTimeMs;
    double executionTimeMs;
    double dataTransferTimeMs;
    double throughput;
    double speedup;
    bool sortedOk;
};

struct CudaResult
{
    std::string type;
    int size;

    int blocks;
    int threadsPerBlock;
    double computationTimeMs;
    double executionTimeMs;
    double dataTransferTimeMs;
    double throughput;
    bool sortedOk;
};

// Print table header for CPU results
inline void printCpuResultHeader()
{
    std::cout
        << std::left
        << std::setw(15) << "Type"
        << std::setw(12) << "Size"
        << std::setw(15) << "Threads"
        << std::setw(15) << "Computation(ms)"
        << std::setw(15) << "Execution(ms)"
        << std::setw(15) << "Data Transfer(ms)"
        << std::setw(18) << "Throughput"
        << std::setw(12) << "Speedup"
        << std::setw(10) << "Sorted"
        << '\n';

    std::cout << std::string(127, '-') << '\n';
}

// Print one CPU result row
inline void printCpuResult(const CpuResult &result)
{
    std::cout
        << std::left
        << std::setw(15) << result.type
        << std::setw(12) << result.size
        << std::setw(15) << result.threadCount
        << std::setw(15) << std::fixed << std::setprecision(3) << result.computationTimeMs
        << std::setw(15) << std::fixed << std::setprecision(3) << result.executionTimeMs
        << std::setw(15) << std::fixed << std::setprecision(3) << result.dataTransferTimeMs
        << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput
        << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup
        << std::setw(10) << (result.sortedOk ? "Yes" : "No")
        << '\n';
}

inline void writeCpuCsvHeader(const std::string &filename)
{
    std::ofstream file(filename);

    file << "Type,"
         << "Size,"
         << "Threads,"
         << "ComputationTimeMs,"
         << "ExecutionTimeMs,"
         << "DataTransferTimeMs,"
         << "Throughput,"
         << "Speedup,"
         << "Sorted"
         << '\n';
}

inline void appendCpuResultToCsv(const std::string &filename, const CpuResult &result)
{
    std::ofstream file(filename, std::ios::app);

    file << result.type << ','
         << result.size << ','
         << result.threadCount << ','
         << std::fixed << std::setprecision(3) << result.computationTimeMs << ','
         << std::fixed << std::setprecision(3) << result.executionTimeMs << ','
         << std::fixed << std::setprecision(3) << result.dataTransferTimeMs << ','
         << std::fixed << std::setprecision(2) << result.throughput << ','
         << std::fixed << std::setprecision(2) << result.speedup << ','
         << (result.sortedOk ? "Yes" : "No")
         << '\n';
}

// Print table header for CUDA results
inline void printCudaResultHeader()
{
    std::cout
        << std::left
        << std::setw(15) << "Type"
        << std::setw(12) << "Size"
        << std::setw(12) << "Blocks"
        << std::setw(15) << "Threads/Block"
        << std::setw(15) << "Computation(ms)"
        << std::setw(15) << "Execution(ms)"
        << std::setw(15) << "Data Transfer(ms)"
        << std::setw(18) << "Throughput"
        << std::setw(10) << "Sorted"
        << '\n';

    std::cout << std::string(127, '-') << '\n';
}

// Print one CUDA result row
inline void printCudaResult(const CudaResult &result)
{
    std::cout
        << std::left
        << std::setw(15) << result.type
        << std::setw(12) << result.size
        << std::setw(12) << result.blocks
        << std::setw(15) << result.threadsPerBlock
        << std::setw(15) << std::fixed << std::setprecision(3) << result.computationTimeMs
        << std::setw(15) << std::fixed << std::setprecision(3) << result.executionTimeMs
        << std::setw(15) << std::fixed << std::setprecision(3) << result.dataTransferTimeMs
        << std::setw(18) << std::fixed << std::setprecision(2) << result.throughput
        << std::setw(10) << (result.sortedOk ? "Yes" : "No")
        << '\n';
}

inline void writeCudaCsvHeader(const std::string &filename)
{
    std::ofstream file(filename);

    file << "Type,"
         << "Size,"
         << "Blocks,"
         << "ThreadsPerBlock,"
         << "ComputationTimeMs,"
         << "ExecutionTimeMs,"
         << "DataTransferTimeMs,"
         << "Throughput,"
         << "Sorted"
         << '\n';
}

inline void appendCudaResultToCsv(const std::string &filename, const CudaResult &result)
{
    std::ofstream file(filename, std::ios::app);

    file << result.type << ','
         << result.size << ','
         << result.blocks << ','
         << result.threadsPerBlock << ','
         << std::fixed << std::setprecision(3) << result.computationTimeMs << ','
         << std::fixed << std::setprecision(3) << result.executionTimeMs << ','
         << std::fixed << std::setprecision(3) << result.dataTransferTimeMs << ','
         << std::fixed << std::setprecision(2) << result.throughput << ','
         << (result.sortedOk ? "Yes" : "No")
         << '\n';
}