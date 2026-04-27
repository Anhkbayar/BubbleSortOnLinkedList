#pragma once
#include <atomic>
#include <cstddef>
#include <thread>

class ReusableBarrier
{
private:
    const std::size_t threshold;
    std::atomic<std::size_t> count;
    std::atomic<std::size_t> generation;

    template <typename Completion>
    void waitImpl(Completion completion)
    {
        const std::size_t gen = generation.load(std::memory_order_acquire);

        if (count.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            // Last arriving thread runs optional phase-completion work first.
            completion();
            count.store(threshold, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_acq_rel);
        }
        else
        {
            // Spin-yield avoids the heavier mutex/condition_variable wakeup cost.
            while (generation.load(std::memory_order_acquire) == gen)
            {
                std::this_thread::yield();
            }
        }
    }

public:
    explicit ReusableBarrier(std::size_t num_threads)
        : threshold(num_threads), count(num_threads), generation(0) {}

    // Blocks until all worker threads reach the same phase boundary.
    // The generation counter lets the same barrier be reused every round.
    void wait()
    {
        waitImpl([] {});
    }

    // Same barrier wait, but the last arriving thread runs completion before
    // the waiting threads are released.
    template <typename Completion>
    void wait(Completion completion)
    {
        waitImpl(completion);
    }
};
