#include "../linked_list.h"
#include "../results.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Multithreaded odd-even bubble sort implementation from MultithreadBubbleSort.cpp.
void parallelBubbleSortLinkedList(Node *head, int numThreads);

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

int main()
{
    // Assignment-required input sizes: 10k, 100k, and 200k linked-list nodes.
    const std::vector<int> sizes = {10000, 100000, 200000};
    const std::vector<int> threadCounts = buildThreadCounts();
    const std::string csvFilename = "multithread_results.csv";
    std::vector<CpuResult> results;

    std::cout << "--- Multithread Linked list Sort ---\n\n";
    writeCpuCsvHeader(csvFilename);

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
            const double computationTimeMs = timeMs;
            const double executionTimeMs = timeMs;
            const double dataTransferTimeMs = 0.0;
            const double throughput =
                (timeMs > 0.0) ? static_cast<double>(n) / (timeMs / 1000.0) : 0.0;
            const bool sortedOk = isSorted(head, n);

            if (threadCount == 1)
            {
                baselineTimeMs = timeMs;
            }

            const double speedup =
                (threadCount == 1 || baselineTimeMs <= 0.0) ? 1.0 : baselineTimeMs / timeMs;

            CpuResult result{"Multithread", n, threadCount, computationTimeMs,
                             executionTimeMs, dataTransferTimeMs, throughput,
                             speedup, sortedOk};
            results.push_back(result);
            appendCpuResultToCsv(csvFilename, result);

            std::cout << "Run Result\n";
            printCpuResultHeader();
            printCpuResult(result);
            std::cout << "\n";

            deleteList(head);
        }
    }

    std::cout << "\nSummary\n";
    printCpuResultHeader();
    for (const CpuResult &result : results)
    {
        printCpuResult(result);
    }

    return 0;
}
