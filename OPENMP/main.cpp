#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <string>
#include "../linked_list.h"
#include "../results.h"

using namespace std;

/**
 * sequentialBubbleSort - Дараалсан (Sequential) бөмбөлгөн эрэмбэлэлт
 */
CpuResult sequentialBubbleSort(Node* head, int n) {
    auto t0 = chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        Node* cur = head;
        for (int j = 0; j < n - i - 1; ++j) {
            if (cur->data > cur->next->data) {
                swap(cur->data, cur->next->data);
                swapped = true;
            }
            cur = cur->next;
        }
        if (!swapped) break;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double time = chrono::duration<double, milli>(t1 - t0).count();
    bool ok = isSorted(head, n);
    double throughput = (time > 0 ? n / (time / 1000.0) : 0);

    return {"Sequential", n, 1, time, time, 0.0, throughput, 1.0, ok};
}

/**
 * openmpBubbleSort - OpenMP ашигласан Параллел бөмбөлгөн эрэмбэлэлт
 */
CpuResult openmpBubbleSort(Node* head, int n, int num_threads, double baselineTime) {
    auto ts0 = chrono::high_resolution_clock::now();
    vector<Node*> nodes(n);
    Node* cur = head;
    for (int i = 0; i < n; ++i) { nodes[i] = cur; cur = cur->next; }
    auto ts1 = chrono::high_resolution_clock::now();
    double transfer_ms = chrono::duration<double, milli>(ts1 - ts0).count();

    auto t0 = chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(num_threads)
    {
        for (int phase = 0; phase < n; ++phase) {
            int start_idx = (phase % 2 == 0) ? 0 : 1;
            #pragma omp for nowait
            for (int i = start_idx; i < n - 1; i += 2) {
                if (nodes[i]->data > nodes[i + 1]->data) {
                    swap(nodes[i]->data, nodes[i + 1]->data);
                }
            }
            #pragma omp barrier
        }
    }
    auto t1 = chrono::high_resolution_clock::now();
    double comp_ms = chrono::duration<double, milli>(t1 - t0).count();
    bool ok = isSorted(head, n);
    double throughput = (comp_ms > 0 ? n / (comp_ms / 1000.0) : 0);
    double speedup = (baselineTime > 0) ? baselineTime / comp_ms : 1.0;

    return {"OpenMP", n, num_threads, comp_ms, comp_ms + transfer_ms, transfer_ms, throughput, speedup, ok};
}

int main() {
    vector<int> sizes = {10000, 100000, 200000}; 
    int max_threads = omp_get_max_threads();
    vector<int> threads_to_test;
    if (max_threads >= 2) threads_to_test.push_back(2);
    if (max_threads >= 4) threads_to_test.push_back(4);
    if (max_threads >= 8) threads_to_test.push_back(8);
    if (find(threads_to_test.begin(), threads_to_test.end(), max_threads) == threads_to_test.end()) {
        threads_to_test.push_back(max_threads);
    }

    string csv_filename = "openmp_results.csv";
    writeCpuCsvHeader(csv_filename);

    cout << "================================================================\n";
    cout << "  F.CSM306 - OpenMP Bubble Sort on Linked List Benchmark\n";
    cout << "  Хэрэгжүүлсэн: Оновчтой Odd-Even Transposition Sort\n";
    cout << "================================================================\n\n";

    for (int run = 1; run <= 3; ++run) {
        cout << ">>> Trial #" << run << " <<<\n";
        printCpuResultHeader();

        for (int n : sizes) {
            Node* head_seq = createList(n);
            CpuResult res_seq = sequentialBubbleSort(head_seq, n);
            printCpuResult(res_seq);
            appendCpuResultToCsv(csv_filename, res_seq);
            deleteList(head_seq);

            for (int t : threads_to_test) {
                Node* head_omp = createList(n);
                CpuResult res_omp = openmpBubbleSort(head_omp, n, t, res_seq.computationTimeMs);
                printCpuResult(res_omp);
                appendCpuResultToCsv(csv_filename, res_omp);
                deleteList(head_omp);
            }
            cout << string(127, '-') << "\n";
        }
        cout << endl;
    }

    return 0;
}