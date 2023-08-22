    #include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include <string.h>
#include <time.h>

int* Get_List_Index(Matrix* this, int index)
{
    int* list = malloc(this->size_shape*sizeof(int));
    int tmp;
    for (int i = this->size_shape-1; i!=-1 ; i--)
    {
        tmp = this->shape[i];
        list[i] = index%tmp;
        index = index/tmp;
    }
    return list;
}
int Get_Index(Matrix* m,int* index_liste)
{
    int res = 0;
    int acc = 1;
    for (int i = 0; i!=m->size_shape; i++)
    {
        res += acc*index_liste[i];
        acc*=m->shape[i];
    }
    return res;
}
int Calculate_Index(Matrix* this, int* list_index_res, int size_shape)
{
    int index = 0;
    int acc = 1;
    for (int i = 0; i!=this->size_shape; i++)
    {
        int index_res = list_index_res[size_shape-1-i];
        int index_m = this->size_shape-i-1;
        index += (index_res % this->shape[index_m]) * acc;
        acc *= this->shape[index_m];
    }
    return index;
}
int Get_Swap_Index (int index, Matrix old, Matrix* new, int p1, int p2)
{
    /*
    Return the index of the new matrix
    @param
    INDEX: The index of the old matrix
    OLD: The old matrix
    NEW: The new matrix
    P1: The first position
    P2: The second position
    @return
    The index of the new matrix
    */
    int* index_liste = malloc(old.size_shape*sizeof(int));
    int size_shape = old.size_shape;
    for (int i = size_shape-1; i!=-1 ; i--)
    {
        int tmp = old.shape[i];
        index_liste[i] = index%tmp;
        index = index/tmp;
    }
    int tmp = index_liste[p1];
    index_liste[p1] = index_liste[p2];
    index_liste[p2] = tmp;
    int res = 0;
    int acc = 1;
    for (int i = size_shape-1; i!=-1 ; i--)
    {
        res += acc*index_liste[i];
        acc*=new->shape[i];
    }
    free(index_liste);
    return res;
}
int Shape_Are_Equal (int* shape_1, int* shape_2, int size_shape_1, int size_shape_2)
{
    if (size_shape_1 != size_shape_2)
    {
        return 0;
    }
    for (int i = 0; i!=size_shape_1; i++)
    {
        if (shape_1[i] != shape_2[i])
        {
            return 0;
        }
    }
    return 1;
}