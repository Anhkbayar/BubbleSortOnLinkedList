#include "../linked_list.h"
#include "ReusableBarrier.h"

#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>
#include <utility>

// Convert linked list into an indexable vector of node pointers
static std::vector<Node *> buildNodeVector(Node *head)
{
    std::vector<Node *> nodes;
    for (Node *p = head; p != nullptr; p = p->next)
    {
        nodes.push_back(p);
    }
    return nodes;
}

void parallelBubbleSortLinkedList(Node *head, int numThreads)
{
    if (head == nullptr || head->next == nullptr)
        return;

    std::vector<Node *> nodes = buildNodeVector(head);
    const int n = static_cast<int>(nodes.size());

    if (n < 2)
        return;
    if (numThreads <= 0)
        numThreads = 1;

    // In one phase, maximum useful parallel comparisons is n/2
    numThreads = std::min(numThreads, std::max(1, n / 2));

    ReusableBarrier barrier(numThreads);
    std::atomic<bool> swapped(false);
    std::atomic<bool> done(false);

    auto processPhase = [&](int tid, int startIndex)
    {
        int pairCount = (n - startIndex) / 2;

        int beginPair = (tid * pairCount) / numThreads;
        int endPair = ((tid + 1) * pairCount) / numThreads;

        for (int p = beginPair; p < endPair; ++p)
        {
            int i = startIndex + 2 * p;
            if (i + 1 < n && nodes[i]->data > nodes[i + 1]->data)
            {
                std::swap(nodes[i]->data, nodes[i + 1]->data);
                swapped.store(true, std::memory_order_relaxed);
            }
        }
    };

    auto worker = [&](int tid)
    {
        for (int round = 0; round < n; ++round)
        {
            if (tid == 0)
            {
                swapped.store(false, std::memory_order_relaxed);
            }
            barrier.wait();

            // Even phase: (0,1), (2,3), (4,5), ...
            processPhase(tid, 0);
            barrier.wait();

            // Odd phase: (1,2), (3,4), (5,6), ...
            processPhase(tid, 1);
            barrier.wait();

            if (tid == 0)
            {
                done.store(!swapped.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
            }
            barrier.wait();

            if (done.load(std::memory_order_relaxed))
            {
                break;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (int t = 0; t < numThreads; ++t)
    {
        threads.emplace_back(worker, t);
    }

    for (auto &th : threads)
    {
        th.join();
    }
}
