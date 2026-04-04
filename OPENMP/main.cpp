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

using namespace std;

// ─────────────────────────────────────────────────────────────────────────────
//  Node  –  singly linked list node
// ─────────────────────────────────────────────────────────────────────────────
struct Node {
    int   data;
    Node* next;
};

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
//  createList
//  Builds a linked list of n random integers.
//  srand(42) is reset on EVERY call so all versions receive identical data.
// ─────────────────────────────────────────────────────────────────────────────
Node* createList(int n) {
    if (n <= 0) return nullptr;
    srand(42);
    Node* head = new Node{rand() % 1000000, nullptr};
    Node* cur  = head;
    for (int i = 1; i < n; ++i) {
        cur->next = new Node{rand() % 1000000, nullptr};
        cur = cur->next;
    }
    return head;
}

// ─────────────────────────────────────────────────────────────────────────────
//  deleteList  –  frees every node
// ─────────────────────────────────────────────────────────────────────────────
void deleteList(Node* head) {
    while (head) {
        Node* nxt = head->next;
        delete head;
        head = nxt;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  isSorted  –  O(n) ascending-order correctness check
// ─────────────────────────────────────────────────────────────────────────────
bool isSorted(Node* head, int n) {
    if (!head) return (n == 0);
    Node* cur = head;
    int   cnt = 1;
    while (cur->next) {
        if (cur->data > cur->next->data) return false;
        cur = cur->next;
        ++cnt;
    }
    return (cnt == n);
}

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
//
//  Optimisation:
//    After i outer passes the last i elements are in their final positions,
//    so the inner loop shrinks each iteration.
//    If a complete inner pass produces zero swaps the list is already sorted;
//    the outer loop exits early to avoid wasted work.
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
//
//  Why not standard Bubble Sort with OpenMP?
//  ─────────────────────────────────────────
//  In standard bubble sort the inner loop has a sequential dependency:
//  iteration j+1 reads a node that iteration j may have just modified.
//  This prevents safe parallel execution of the inner loop.
//
//  Odd-Even Transposition Sort removes that dependency by splitting each
//  pass into two phases of NON-OVERLAPPING pairs:
//
//    Even phase:  compare (0,1), (2,3), (4,5), ...
//    Odd  phase:  compare (1,2), (3,4), (5,6), ...
//
//  Within a single phase all pairs are disjoint, so every pair can be
//  processed in parallel with no race condition.  N alternating phases
//  guarantee a fully sorted result (proven via the 0-1 principle).
//
//  Random-access trick for linked lists:
//  ──────────────────────────────────────
//  A singly linked list provides only O(n) sequential access.  To let each
//  OpenMP thread jump directly to its assigned pair we cache all node
//  pointers in a vector<Node*> once before the sort loop starts.
//  Threads index the vector in O(1) and swap the DATA fields of the
//  corresponding nodes.  The list's pointer structure is never altered.
//
//  OpenMP features used:
//    omp_set_num_threads(t)                 – sets the thread-pool size
//    #pragma omp parallel for               – distributes loop iterations
//    reduction(+: phase_comps, phase_swaps) – race-free counter accumulation
//
//  Early-exit optimisation:
//    If an EVEN phase produces zero swaps the array is fully sorted and the
//    outer loop terminates (analogous to the sequential early-exit flag).
//    Odd phases alone cannot confirm global sorted order, so early exit is
//    only triggered after a swap-free even phase.
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

        // Even phase: first pair at index 0  → pairs (0,1),(2,3),...
        // Odd  phase: first pair at index 1  → pairs (1,2),(3,4),...
        int start_idx = (phase % 2 == 0) ? 0 : 1;

        // Each loop iteration handles one disjoint pair (i, i+1).
        // Stepping by 2 ensures no two threads touch the same node.
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

    // !! WARNING – 1 000 000 elements !!
    // Sequential bubble sort is O(n²).  At 1M elements the sequential version
    // can take several HOURS.  Uncomment 1000000 only when intentional.
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

    // ── Build sequential baseline map for SpeedUp calculation ─────────────────
    // Sequential SpeedUp is always defined as 1.0x by convention.
    map<int, double> seq_base;
    for (const auto& r : results)
        if (r.version == "Sequential") seq_base[r.n] = r.time_ms;

    // ── Main comparison table ──────────────────────────────────────────────────
    cout << "\n";
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";
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

        // Sequential SpeedUp is always 1.0x by definition
        double avg_sp = (ver == "Sequential") ? 1.0
                      : (sp_cnt > 0)          ? sp_sum / sp_cnt
                      :                          0.0;

        cout << "| " << left  << setw(15) << ver
             << " | " << right << setw(7)  << thr << " |";

        // 10k column (8 wide) / 100k column (9 wide) / 1M column (8 wide)
        int widths[3] = {8, 9, 8};
        for (int k = 0; k < 3; ++k) {
            if (t[k] > 0.0)
                cout << setw(widths[k]) << fixed << setprecision(2) << t[k] << " |";
            else
                cout << setw(widths[k] - 2) << "N/A" << "   |";
        }
        cout << setw(6) << fixed << setprecision(2) << avg_sp << "x |\n";
    };

    printRow("Sequential", 1);
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";
    for (int t : threads_to_test) printRow("OpenMP", t);
    cout << "+-----------------+---------+----------+-----------+---------+---------+\n";

    // ── Detailed metrics table ─────────────────────────────────────────────────
    cout << "\n";
    cout << "+-------------+---------+------+--------------+--------------+--------------------+\n";
    cout << "| Version     | Threads | Size |  Comparisons |        Swaps | Throughput(elem/s) |\n";
    cout << "+-------------+---------+------+--------------+--------------+--------------------+\n";

    for (const auto& r : results) {
        double thr = (r.time_ms > 0.0) ? r.n / (r.time_ms / 1000.0) : 0.0;
        cout << "| " << left  << setw(11) << r.version
             << " | " << right << setw(7)  << r.threads
             << " | " << setw(4)  << sizeLabel(r.n)
             << " | " << setw(12) << r.comparisons
             << " | " << setw(12) << r.swaps
             << " | " << setw(18) << fixed << setprecision(0) << thr
             << " |\n";
    }
    cout << "+-------------+---------+------+--------------+--------------+--------------------+\n";

    cout << "\nNotes:\n";
    cout << "  SpeedUp = Sequential_time / OpenMP_time  (averaged across tested sizes).\n";
    cout << "  SpeedUp < 1.0x means OpenMP was SLOWER than sequential.\n";
    cout << "  This is expected for small n where thread overhead dominates.\n";
    cout << "  OpenMP version uses Odd-Even Transposition Sort: disjoint adjacent-pair\n";
    cout << "  phases (even: (0,1),(2,3),...  odd: (1,2),(3,4),...) allow all pairs\n";
    cout << "  within a phase to be processed in parallel without data races.\n";

    return 0;
}