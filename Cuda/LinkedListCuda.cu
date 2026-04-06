#include <chrono>
#include <iostream>

#include "../linked_list.h"

//Array to list conversion
int *listToArray(Node *head, int n)
{
    int *arr = new int[n];
    Node *p = head;
    for (int i = 0; i < n; i++, p = p->next)
    {
        arr[i] = p->data;
    }
    return arr;
}

void arrayToList(Node *head, const int *arr, int n)
{
    Node *p = head;
    for (int i = 0; i < n; i++, p = p->next)
    {
        p->data = arr[i];
    }
}

//Kernel
__global__ void oddEvenKernel(int* arr, int n, int phase){
    
}