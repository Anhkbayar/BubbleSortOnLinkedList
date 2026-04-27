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

using namespace std;

/**
 * Result - Туршилтын үр дүнг хадгалах бүтэц
 * Огноо, хурд, харьцуулалт, солилцоо зэрэг үзүүлэлтүүдийг агуулна.
 */
struct Result {
    string    version;
    int       threads;
    int       n;
    double    time_ms;
    long long comparisons;
    long long swaps;
};

/**
 * printStats - Тухайн туршилтын үр дүнг дэлгэцэнд хэвлэх функц
 */
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

/**
 * sequentialBubbleSort - Дараалсан (Sequential) бөмбөлгөн эрэмбэлэлт
 * Хугацааны нарийн төвөгтэй байдал: O(n^2)
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
        if (!swapped) break; // Жагсаалт эрэмбэлэгдсэн бол эрт гарах
    }

    auto t1 = chrono::high_resolution_clock::now();
    if (!isSorted(head, n)) cerr << "[ERROR] Sequential result is NOT sorted!\n";

    return {"Sequential", 1, n, chrono::duration<double, milli>(t1 - t0).count(), comparisons, swaps};
}

/**
 * openmpBubbleSort - OpenMP ашигласан Параллел бөмбөлгөн эрэмбэлэлт
 * "Odd-Even Transposition" алгоритмыг ашиглан параллелчилсан.
 * 
 * Оновчлол:
 * 1. Persistent Parallel Region: Thread-үүдийг нэг удаа үүсгэж, fork-join overhead-ийг багасгасан.
 * 2. Pointer Caching: Linked List-ийн элементүүдийн хаягийг vector-т хадгалж, санах ойн хандалтыг (O(1)) хурдасгасан.
 * 3. Reduction: Comparisons болон Swaps-ийг thread-safe байдлаар тоолсон.
 */
Result openmpBubbleSort(Node* head, int n, int num_threads) {
    long long total_comparisons = 0;
    long long total_swaps       = 0;

    // Жагсаалтын зангилаануудын хаягийг кэшлэх (Санах ойн локал чанарыг сайжруулах)
    vector<Node*> nodes(n);
    Node* cur = head;
    for (int i = 0; i < n; ++i) { nodes[i] = cur; cur = cur->next; }

    auto t0 = chrono::high_resolution_clock::now();

    // Параллел мужийг нэг удаа үүсгэх
    #pragma omp parallel num_threads(num_threads) reduction(+:total_comparisons, total_swaps)
    {
        for (int phase = 0; phase < n; ++phase) {
            int start_idx = (phase % 2 == 0) ? 0 : 1;

            // Хөрш элементүүдийг зэрэг харьцуулж солих
            #pragma omp for nowait
            for (int i = start_idx; i < n - 1; i += 2) {
                ++total_comparisons;
                if (nodes[i]->data > nodes[i + 1]->data) {
                    swap(nodes[i]->data, nodes[i + 1]->data);
                    ++total_swaps;
                }
            }

            // Үе бүрийн дараа thread-үүдийг синхрончлох (Lockstep)
            #pragma omp barrier
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    if (!isSorted(head, n)) cerr << "[ERROR] OpenMP result is NOT sorted!\n";

    return {"OpenMP", num_threads, n, chrono::duration<double, milli>(t1 - t0).count(), total_comparisons, total_swaps};
}

/**
 * displaySummaryTable - Туршилтын нэгдсэн үр дүнг хүснэгтээр харуулах
 */
void displaySummaryTable(const vector<Result>& results, const vector<int>& sizes) {
    map<int, double> seq_base;
    for (const auto& r : results) if (r.version == "Sequential") seq_base[r.n] = r.time_ms;

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
            for (const auto& r : results) if (r.version == ver && r.threads == thr && r.n == n) time = r.time_ms;
            if (time > 0) {
                cout << setw(10) << fixed << setprecision(2) << time << " |";
                total_su += seq_base[n] / time;
                count++;
            } else cout << "    N/A    |";
        }
        cout << setw(8) << fixed << setprecision(2) << (count ? total_su/count : 0.0) << "x |\n";
    };

    printRow("Sequential", 1);
    cout << "+-----------------+---------+";
    for (size_t i=0; i<sizes.size(); ++i) cout << "-----------+";
    cout << "---------+\n";

    vector<int> thr_tested = {2, 4, 8};
    for (int t : thr_tested) printRow("OpenMP", t);
    cout << "+-----------------+---------+";
    for (size_t i=0; i<sizes.size(); ++i) cout << "-----------+";
    cout << "---------+\n";
}

int main() {
    // Даалгаврын шаардлагын дагуу 10k, 100k, 1M хэмжээтэй өгөгдөл дээр турших
    vector<int> sizes = {10000, 100000, 1000000}; 
    
    // Системийн боломжит thread-үүдийг тодорхойлох
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
    cout << "  Системийн нийт thread: " << max_threads << "\n";
    cout << "================================================================\n";

    for (int n : sizes) {
        cout << "\n[Dataset: " << n << " elements]\n";
        
        Node* head_seq = createList(n);
        cout << " -> Running Sequential...\n";
        Result r_seq = sequentialBubbleSort(head_seq, n);
        results.push_back(r_seq);
        printStats(r_seq);
        deleteList(head_seq);

        for (int t : threads_to_test) {
            Node* head_omp = createList(n);
            cout << " -> Running OpenMP (" << t << " threads)...\n";
            Result r_omp = openmpBubbleSort(head_omp, n, t);
            results.push_back(r_omp);
            printStats(r_omp);
            deleteList(head_omp);
        }
    }

    displaySummaryTable(results, sizes);
    
    cout << "\n* 1M элементийг турших бол main() доторх 'sizes' хэсгийг өөрчилнө үү.\n";
    cout << "* Бөмбөлгөн эрэмбэлэлт нь O(n^2) тул 1M дээр маш их хугацаа авдаг.\n";

    return 0;
}