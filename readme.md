```
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
```
