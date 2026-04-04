#pragma once
#include <cstdlib>   // rand, srand
#include <iostream>  // print

//Node
struct Node {
    int   val;
    Node* next;
};

//Factory
inline Node* createList(int n) {
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

//Helpers
inline void printList(Node* head, int limit = 20) {
    int cnt = 0;
    for (Node* p = head; p && cnt < limit; p = p->next, ++cnt)
        std::cout << p->val << (p->next && cnt + 1 < limit ? " -> " : "");
    std::cout << (limit && head ? " ...\n" : "\n");
}

inline void freeList(Node* head) {
    while (head) { Node* t = head->next; delete head; head = t; }
}

inline int listLength(Node* head) {
    int n = 0; for (Node* p = head; p; p = p->next) ++n; return n;
}