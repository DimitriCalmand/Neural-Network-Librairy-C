#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include "Copy.h"
#include <string.h>
#include <time.h>

Matrix* __Concat_Matrix(Matrix m1, Matrix m2, int axis)
{
    if (axis < 0)
    {
        axis = m1.size_shape + axis;
    }
    if (axis > m1.size_shape)
    {
        errx(1, "Axis out of range in concatenate");
    }
    if (m1.size_shape != m2.size_shape)
    {
        errx(1, "Incompatible shapes in concatenate");
    }
    for (int i = 0; i!=m1.size_shape; i++)
    {
        if (i!=axis && m1.shape[i]!=m2.shape[i])
        {
            errx(1, "Incompatible shapes in concatenate (axis %d)", i);
        }
    }
    int* shape = malloc(m1.size_shape*sizeof(int));
    for (int i = 0; i!=m1.size_shape; i++)
    {
        shape[i] = m1.shape[i];
    }
    shape[axis] += m2.shape[axis];
    Matrix* res = Init_Matrix(shape, m1.size_shape, "null");
    int step_1 = 1;
    int step_2 = 1;
    int nb_iter = 1;
    for (int i = m1.size_shape-1; i>axis-1; i--)
    {
        step_1 *= m1.shape[i];
        step_2 *= m2.shape[i];
    }
    for (int i = 0; i<axis; i++)
    {
        nb_iter *= m1.shape[i];
    }
    int index = 0;
    int j =0;
    while (j!=nb_iter)
    {
        for (int i = 0; i!= step_1; i++)
        {
            res->array[index++] = m1.array[j*step_1+i];
        }
        for (int i = 0; i!= step_2; i++)
        {
            res->array[index++] = m2.array[j*step_2+i];
        }
        j++;
    }
    free(shape);
    return res;
}
void Concat_Matrix(Matrix* m1, Matrix* m2, int axis)
{
    Matrix* res = __Concat_Matrix(*m1, *m2, axis);
    m1->Copy(m1, res);
    res->Free(res);
}