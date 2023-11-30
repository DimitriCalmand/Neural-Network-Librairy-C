#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include "../Matrix.h"
int* Get_List_Index(Matrix* this, int index);
int Get_Index(Matrix* m,int* index_liste);
int Calculate_Index(Matrix* this, int* list_index_res, int size_shape);
int Get_Swap_Index (int index, Matrix old, Matrix* new, int p1, int p2);
int Shape_Are_Equal (int* shape_1, int* shape_2, int size_shape_1, int size_shape_2);
#endif