#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <map>
#include <string>
#include "../linked_list.h"

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
//  Result  –  one benchmark run's collected metrics
// ─────────────────────────────────────────────────────────────────────────────
struct Result {
    string    version;
    int       threads;
    int       n;
    double    time_ms;
    long long comparisons;
    long long swaps;
};

// ─────────────────────────────────────────────────────────────────────────────
//  printStats
//  Prints timing, comparisons, swaps, and throughput for a single Result.
//  Called after every sort run to provide immediate per-run feedback.
// ─────────────────────────────────────────────────────────────────────────────
void printStats(const Result& r) {
    double throughput = (r.time_ms > 0.0) ? r.n / (r.time_ms / 1000.0) : 0.0;

    cout << "  ┌─ Version     : " << r.version << "\n";
    cout << "  │  Threads     : " << r.threads << "\n";
    cout << "  │  Size        : " << r.n << " elements\n";
    cout << fixed << setprecision(2);
    cout << "  │  Time        : " << r.time_ms << " ms\n";
    cout << "  │  Comparisons : " << r.comparisons << "\n";
    cout << "  │  Swaps       : " << r.swaps << "\n";
    cout << fixed << setprecision(0);
    cout << "  └─ Throughput  : " << throughput << " elem/s\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  sequentialBubbleSort
//
//  Classic O(n²) bubble sort adapted for a singly linked list.
//  Only DATA values are swapped – pointer structure is never touched.
// ─────────────────────────────────────────────────────────────────────────────
Result sequentialBubbleSort(Node* head, int n) {
    long long comparisons = 0;
    long long swaps       = 0;

    auto t0 = chrono::high_resolution_clock::now();

    bool swapped;
    for (int i = 0; i < n - 1; ++i) {
        swapped   = false;
        Node* cur = head;

        for (int j = 0; j < n - i - 1; ++j) {
            ++comparisons;
            if (cur->data > cur->next->data) {
                swap(cur->data, cur->next->data);
                ++swaps;
                swapped = true;
            }
            cur = cur->next;
        }

        if (!swapped) break; // Early exit: no swap means sorted
    }

    auto t1 = chrono::high_resolution_clock::now();

    if (!isSorted(head, n))
        cerr << "[ERROR] Sequential result is NOT sorted!\n";

    return {"Sequential", 1, n,
            chrono::duration<double, milli>(t1 - t0).count(),
            comparisons, swaps};
}

// ─────────────────────────────────────────────────────────────────────────────
//  openmpBubbleSort
//
//  Odd-Even Transposition Sort parallelised with OpenMP.
// ─────────────────────────────────────────────────────────────────────────────
Result openmpBubbleSort(Node* head, int n, int num_threads) {
    long long total_comparisons = 0;
    long long total_swaps       = 0;

    // Cache node pointers for O(1) random access inside OpenMP threads
    vector<Node*> nodes(n);
    {
        Node* cur = head;
        for (int i = 0; i < n; ++i) { nodes[i] = cur; cur = cur->next; }
    }

    omp_set_num_threads(num_threads);

    bool any_swap = true; // Tracks whether early exit is possible

    auto t0 = chrono::high_resolution_clock::now();

    for (int phase = 0; phase < n && any_swap; ++phase) {

        long long phase_comps = 0;
        long long phase_swaps = 0;

        int start_idx = (phase % 2 == 0) ? 0 : 1;

        #pragma omp parallel for reduction(+:phase_comps, phase_swaps)
        for (int i = start_idx; i < n - 1; i += 2) {
            ++phase_comps;
            if (nodes[i]->data > nodes[i + 1]->data) {
                swap(nodes[i]->data, nodes[i + 1]->data);
                ++phase_swaps;
            }
        }

        total_comparisons += phase_comps;
        total_swaps       += phase_swaps;

        // Early exit: only valid to check after a swap-free EVEN phase
        if (phase_swaps == 0 && (phase % 2 == 0))
            any_swap = false;
    }

    auto t1 = chrono::high_resolution_clock::now();

    if (!isSorted(head, n))
        cerr << "[ERROR] OpenMP (" << num_threads << " threads) result is NOT sorted!\n";

    return {"OpenMP", num_threads, n,
            chrono::duration<double, milli>(t1 - t0).count(),
            total_comparisons, total_swaps};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: human-readable size label
// ─────────────────────────────────────────────────────────────────────────────
static string sizeLabel(int n) {
    if (n >= 1000000) return " 1M";
    if (n >= 100000)  return "100k";
    return " 10k";
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    vector<int> sizes           = {10000, 100000 /*, 1000000*/};
    vector<int> threads_to_test = {2, 4, 8};
    int         sz_ref[3]       = {10000, 100000, 1000000};

    vector<Result> results;

    cout << "================================================================\n";
    cout << "  Bubble Sort on Singly Linked List  –  Performance Benchmark\n";
    cout << "  Sequential  vs  OpenMP Odd-Even Transposition Sort\n";
    cout << "================================================================\n";

    for (int n : sizes) {
        cout << "\n[Dataset: " << sizeLabel(n) << " (" << n << " elements)]\n\n";

        // ── Run sequential ────────────────────────────────────────────────────
        {
            Node*  head = createList(n);
            cout << "Running Sequential...\n";
            Result r = sequentialBubbleSort(head, n);
            printStats(r);
            cout << "\n";
            results.push_back(r);
            deleteList(head);
        }

        // ── Run OpenMP with each thread count ─────────────────────────────────
        for (int t : threads_to_test) {
            Node*  head = createList(n);
            cout << "Running OpenMP (" << t << " threads)...\n";
            Result r = openmpBubbleSort(head, n, t);
            printStats(r);
            cout << "\n";
            results.push_back(r);
            deleteList(head);
        }
    }

    map<int, double> seq_base;
    for (const auto& r : results)
        if (r.version == "Sequential") seq_base[r.n] = r.time_ms;

    cout << "\n+-----------------+---------+----------+-----------+---------+---------+\n";
    cout << "| Version         | Threads | 10k (ms) | 100k (ms) | 1M (ms) | SpeedUp |\n";
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";

    auto printRow = [&](const string& ver, int thr) {
        double t[3]  = {0.0, 0.0, 0.0};
        double sp_sum = 0.0;
        int    sp_cnt = 0;

        for (const auto& r : results) {
            if (r.version != ver || r.threads != thr) continue;
            for (int k = 0; k < 3; ++k)
                if (r.n == sz_ref[k]) t[k] = r.time_ms;
            if (seq_base.count(r.n) && r.time_ms > 0.0) {
                sp_sum += seq_base[r.n] / r.time_ms;
                ++sp_cnt;
            }
        }

        double avg_sp = (ver == "Sequential") ? 1.0 : (sp_cnt > 0) ? sp_sum / sp_cnt : 0.0;

        cout << "| " << left  << setw(15) << ver << " | " << right << setw(7)  << thr << " |";

        int widths[3] = {8, 9, 8};
        for (int k = 0; k < 3; ++k) {
            if (t[k] > 0.0) cout << setw(widths[k]) << fixed << setprecision(2) << t[k] << " |";
            else cout << setw(widths[k] - 2) << "N/A" << "   |";
        }
        cout << setw(6) << fixed << setprecision(2) << avg_sp << "x |\n";
    };

    printRow("Sequential", 1);
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";
    for (int t : threads_to_test) printRow("OpenMP", t);
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";

    cout << "\n+-------------+---------+------+--------------+--------------+--------------------+\n";
    cout << "| Version     | Threads | Size |  Comparisons |        Swaps | Throughput(elem/s) |\n";
    cout << "+-------------+---------+------+--------------+--------------+--------------------+\n";

    for (const auto& r : results) {
        double thr = (r.time_ms > 0.0) ? r.n / (r.time_ms / 1000.0) : 0.0;
        cout << "| " << left  << setw(11) << r.version << " | " << right << setw(7)  << r.threads << " | " << setw(4)  << sizeLabel(r.n) << " | " << setw(12) << r.comparisons << " | " << setw(12) << r.swaps << " | " << setw(18) << fixed << setprecision(0) << thr << " |\n";
    }
    cout << "+-------------+---------+------+--------------+--------------+--------------------+\n";

    return 0;
}