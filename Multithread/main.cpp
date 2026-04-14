#include "../linked_list.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

// Function from MultithreadBubbleSort.cpp
void parallelBubbleSortLinkedList(Node *head, int numThreads);

struct Result
{
    int size;
    int threadCount;
    double timeMs;
    double throughput;
    double speedup;
    bool sortedOk;
};

static std::string sizeLabel(int n)
{
    if (n >= 1000000)
    {
        return "1M";
    }
    if (n >= 1000)
    {
        return std::to_string(n / 1000) + "k";
    }
    return std::to_string(n);
}

static std::vector<int> buildThreadCounts()
{
    int hardwareThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (hardwareThreads <= 0)
    {
        hardwareThreads = 4;
    }

    std::vector<int> threadCounts = {1, 2, 4, 8, hardwareThreads};
    std::vector<int> uniqueCounts;

    for (int count : threadCounts)
    {
        if (std::find(uniqueCounts.begin(), uniqueCounts.end(), count) == uniqueCounts.end())
        {
            uniqueCounts.push_back(count);
        }
    }

    return uniqueCounts;
}

static void printRunStats(const Result &result)
{
    std::cout << "  Size        : " << result.size << " (" << sizeLabel(result.size) << ")\n";
    std::cout << "  Threads     : " << result.threadCount << "\n";
    std::cout << "  Sorted      : " << (result.sortedOk ? "true" : "false") << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Time        : " << result.timeMs << " ms\n";
    std::cout << "  Throughput  : " << result.throughput << " elem/s\n";
    std::cout << "  Speedup     : " << result.speedup << "x\n";
}

static void printSummaryTable(const std::vector<Result> &results)
{
    std::cout << "\n+--------+---------+-----------+----------------+---------+--------+\n";
    std::cout << "| Size   | Threads | Time (ms) | Throughput     | Speedup | Sorted |\n";
    std::cout << "+--------+---------+-----------+----------------+---------+--------+\n";

    for (const Result &result : results)
    {
        std::cout << "| " << std::left << std::setw(6) << sizeLabel(result.size)
                  << " | " << std::right << std::setw(7) << result.threadCount
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2) << result.timeMs
                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) << result.throughput
                  << " | " << std::setw(7) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << " | " << std::setw(6) << (result.sortedOk ? "true" : "false") << " |\n";
    }

    std::cout << "+--------+---------+-----------+----------------+---------+--------+\n";
}

int main()
{
    const std::vector<int> sizes = {10000, 100000};
    const std::vector<int> threadCounts = buildThreadCounts();
    std::vector<Result> results;

    std::cout << "===============================================================\n";
    std::cout << "  Bubble Sort on Singly Linked List - Multithread Benchmark\n";
    std::cout << "  C++ Standard Library Threads and Synchronization\n";
    std::cout << "===============================================================\n";

    for (int n : sizes)
    {
        std::cout << "\n[Dataset: " << n << " elements (" << sizeLabel(n) << ")]\n\n";

        double baselineTimeMs = 0.0;

        for (int threadCount : threadCounts)
        {
            Node *head = createList(n);

            auto start = std::chrono::high_resolution_clock::now();
            parallelBubbleSortLinkedList(head, threadCount);
            auto end = std::chrono::high_resolution_clock::now();

            const double timeMs =
                std::chrono::duration<double, std::milli>(end - start).count();
            const double throughput =
                (timeMs > 0.0) ? static_cast<double>(n) / (timeMs / 1000.0) : 0.0;
            const bool sortedOk = isSorted(head, n);

            if (threadCount == 1)
            {
                baselineTimeMs = timeMs;
            }

            const double speedup =
                (threadCount == 1 || baselineTimeMs <= 0.0) ? 1.0 : baselineTimeMs / timeMs;

            Result result{n, threadCount, timeMs, throughput, speedup, sortedOk};
            results.push_back(result);

            std::cout << "Run Result\n";
            printRunStats(result);
            std::cout << "\n";

            deleteList(head);
        }
    }

    printSummaryTable(results);

    return 0;
}
