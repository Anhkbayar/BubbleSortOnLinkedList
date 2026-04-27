#include "../linked_list.h"
#include "ReusableBarrier.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <utility>
#include <vector>

// Build an indexable view of the linked list without copying node values.
// The sort still modifies the original linked-list nodes by swapping their data.
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

    // In one odd-even phase, at most n/2 independent adjacent pairs exist.
    numThreads = std::min(numThreads, std::max(1, n / 2));

    ReusableBarrier barrier(numThreads);
    std::atomic<bool> swapped(false);
    std::atomic<bool> done(false);

    auto processPhase = [&](int tid, int startIndex)
    {
        int pairCount = (n - startIndex) / 2;

        // Split the phase's independent compare-swap pairs evenly among threads.
        int beginPair = (tid * pairCount) / numThreads;
        int endPair = ((tid + 1) * pairCount) / numThreads;

        for (int p = beginPair; p < endPair; ++p)
        {
            int i = startIndex + 2 * p;
            if (i + 1 < n && nodes[i]->data > nodes[i + 1]->data)
            {
                std::swap(nodes[i]->data, nodes[i + 1]->data);
                swapped.store(true);
            }
        }
    };

    auto worker = [&](int tid)
    {
        for (int round = 0; round < n; ++round)
        {
            if (tid == 0)
            {
                swapped.store(false);
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
                // If no thread swapped in this full round, the list is sorted.
                done.store(!swapped.load());
            }
            barrier.wait();

            if (done.load())
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
