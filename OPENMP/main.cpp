#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <map>
#include "../linked_list.h"
#include "../results.h"

using namespace std;

struct Result {
    string    version;
    int       threads;
    int       n;
    double    comp_time_ms;
    double    exec_time_ms;
    double    transfer_time_ms;
    long long comparisons;
    long long swaps;
    bool      sorted;
};

/**
 * sequentialBubbleSort - Дараалсан (Sequential) бөмбөлгөн эрэмбэлэлт
 */
Result sequentialBubbleSort(Node* head, int n) {
    long long comparisons = 0;
    long long swaps       = 0;
    auto t0 = chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
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
        if (!swapped) break;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double time = chrono::duration<double, milli>(t1 - t0).count();
    bool ok = isSorted(head, n);

    return {"Sequential", 1, n, time, time, 0.0, comparisons, swaps, ok};
}

/**
 * openmpBubbleSort - OpenMP ашигласан Параллел бөмбөлгөн эрэмбэлэлт
 */
Result openmpBubbleSort(Node* head, int n, int num_threads) {
    long long total_comparisons = 0;
    long long total_swaps       = 0;

    auto ts0 = chrono::high_resolution_clock::now();
    vector<Node*> nodes(n);
    Node* cur = head;
    for (int i = 0; i < n; ++i) { nodes[i] = cur; cur = cur->next; }
    auto ts1 = chrono::high_resolution_clock::now();
    double transfer_ms = chrono::duration<double, milli>(ts1 - ts0).count();

    auto t0 = chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(num_threads) reduction(+:total_comparisons, total_swaps)
    {
        for (int phase = 0; phase < n; ++phase) {
            int start_idx = (phase % 2 == 0) ? 0 : 1;
            #pragma omp for nowait
            for (int i = start_idx; i < n - 1; i += 2) {
                ++total_comparisons;
                if (nodes[i]->data > nodes[i + 1]->data) {
                    swap(nodes[i]->data, nodes[i + 1]->data);
                    ++total_swaps;
                }
            }
            #pragma omp barrier
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double comp_ms = chrono::duration<double, milli>(t1 - t0).count();
    bool ok = isSorted(head, n);

    return {"OpenMP", num_threads, n, comp_ms, comp_ms + transfer_ms, transfer_ms, total_comparisons, total_swaps, ok};
}

/**
 * displaySummaryTable - Сунгалт болон хурдны харьцуулалт
 */
void displaySummaryTable(const vector<Result>& results, const vector<int>& sizes) {
    map<int, double> seq_base;
    for (const auto& r : results) if (r.version == "Sequential") seq_base[r.n] = r.comp_time_ms;

    cout << "\n+-----------------+---------+";
    for (int n : sizes) {
        string label = (n >= 1000000) ? to_string(n/1000000) + "M" : to_string(n/1000) + "k";
        cout << " " << setw(7) << label << " (ms) |";
    }
    cout << " SpeedUp |\n+-----------------+---------+";
    for (size_t i=0; i<sizes.size(); ++i) cout << "-----------+";
    cout << "---------+\n";

    auto printRow = [&](const string& ver, int thr) {
        cout << "| " << left << setw(15) << ver << " | " << right << setw(7) << thr << " |";
        double total_su = 0;
        int count = 0;
        for (int n : sizes) {
            double time = 0;
            for (const auto& r : results) if (r.version == ver && r.threads == thr && r.n == n) time = r.comp_time_ms;
            if (time > 0) {
                cout << setw(10) << fixed << setprecision(2) << time << " |";
                total_su += seq_base[n] / time;
                count++;
            } else cout << "    N/A    |";
        }
        cout << setw(8) << fixed << setprecision(2) << (count ? total_su/count : 0.0) << "x |\n";
    };

    printRow("Sequential", 1);
    vector<int> thr_tested = {2, 4, 8};
    for (int t : thr_tested) {
        bool exists = false;
        for(const auto& r : results) if(r.threads == t) exists = true;
        if(exists) printRow("OpenMP", t);
    }
    cout << "+-----------------+---------+";
    for (size_t i=0; i<sizes.size(); ++i) cout << "-----------+";
    cout << "---------+\n";
}

int main() {
    vector<int> sizes = {10000, 100000, 1000000}; 
    int max_threads = omp_get_max_threads();
    vector<int> threads_to_test;
    if (max_threads >= 2) threads_to_test.push_back(2);
    if (max_threads >= 4) threads_to_test.push_back(4);
    if (max_threads >= 8) threads_to_test.push_back(8);
    if (find(threads_to_test.begin(), threads_to_test.end(), max_threads) == threads_to_test.end()) {
        threads_to_test.push_back(max_threads);
    }

    vector<Result> results;

    cout << "================================================================\n";
    cout << "  F.CSM306 - OpenMP Bubble Sort on Linked List Benchmark\n";
    cout << "  Хэрэгжүүлсэн: Оновчтой Odd-Even Transposition Sort\n";
    cout << "================================================================\n\n";

    printCpuResultHeader();

    for (int n : sizes) {
        // Sequential
        Node* head_seq = createList(n);
        Result r_seq = sequentialBubbleSort(head_seq, n);
        results.push_back(r_seq);
        
        CpuResult res_seq = {r_seq.version, r_seq.n, r_seq.threads, r_seq.comp_time_ms, r_seq.exec_time_ms, r_seq.transfer_time_ms, 
                             (r_seq.comp_time_ms > 0 ? n/(r_seq.comp_time_ms/1000.0) : 0), 1.0, r_seq.sorted};
        printCpuResult(res_seq);
        deleteList(head_seq);

        // OpenMP
        for (int t : threads_to_test) {
            Node* head_omp = createList(n);
            Result r_omp = openmpBubbleSort(head_omp, n, t);
            results.push_back(r_omp);
            
            double speedup = r_seq.comp_time_ms / r_omp.comp_time_ms;
            CpuResult res_omp = {r_omp.version, r_omp.n, r_omp.threads, r_omp.comp_time_ms, r_omp.exec_time_ms, r_omp.transfer_time_ms,
                                 (r_omp.comp_time_ms > 0 ? n/(r_omp.comp_time_ms/1000.0) : 0), speedup, r_omp.sorted};
            printCpuResult(res_omp);
            deleteList(head_omp);
        }
        cout << string(97, '-') << "\n";
    }


    displaySummaryTable(results, sizes);
    return 0;
}