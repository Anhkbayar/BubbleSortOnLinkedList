#pragma once
#include <cstdlib>   // rand, srand
#include <iostream>  // print

//Node
struct Node {
    int   data;
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

/**
 * deleteList: Frees every node in the linked list.
 */
inline void deleteList(Node* head) {
    while (head) {
        Node* nxt = head->next;
        delete head;
        head = nxt;
    }
}

/**
 * isSorted: Checks if the linked list is correctly sorted in ascending order.
 */
inline bool isSorted(Node* head, int n) {
    if (!head) return (n == 0);
    Node* cur = head;
    int   cnt = 1;
    while (cur->next) {
        if (cur->value > cur->next->value) return false;
        cur = cur->next;
        ++cnt;
    }
    return (cnt == n);
}

/**
 * printList: Utility to print the first 'limit' elements of the list.
 */
inline void printList(Node* head, int limit = 20) {
    int cnt = 0;
    for (Node* p = head; p && cnt < limit; p = p->next, ++cnt)
        std::cout << p->value << (p->next && cnt + 1 < limit ? " -> " : "");
    std::cout << (limit && head ? " ...\n" : "\n");
}

/**
 * listLength: Returns the number of nodes in the list.
 */
inline int listLength(Node* head) {
    int n = 0;
    for (Node* p = head; p; p = p->next) ++n;
    return n;
}