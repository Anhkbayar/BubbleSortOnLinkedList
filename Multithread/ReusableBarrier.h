#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>

class ReusableBarrier
{
private:
    std::mutex mtx;
    std::condition_variable cv;
    std::size_t threshold;
    std::size_t count;
    std::size_t generation;

public:
    explicit ReusableBarrier(std::size_t num_threads)
        : threshold(num_threads), count(num_threads), generation(0) {}

    // Blocks until all worker threads reach the same phase boundary.
    // The generation counter lets the same barrier be reused every round.
    void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        std::size_t gen = generation;

        if (--count == 0)
        {
            // Last arriving thread releases the group and prepares the next use.
            generation++;
            count = threshold;
            cv.notify_all();
        }
        else
        {
            // Predicate protects against spurious wakeups from condition_variable.
            cv.wait(lock, [&]
                    { return gen != generation; });
        }
    }
};
