#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../../Matrix.h"
#include <string.h>
#include <time.h>

VAR Min_Matrix(Matrix* this)
{
    VAR min = 0;
    for (int i = 0; i < this->size_array; i++)
    {
        if (this->array[i] < min)
        {
            min = this->array[i];
        }
    }
    return min;
}

Matrix* __Argmin_2D(Matrix this, int axis, Matrix* res)
{
    if (axis == 0)
    {
        for (int i = 0; i < res->size_array; i++)
        {
            VAR min = 0;
            int index = 0;
            for (int j = 0; j < this.shape[axis]; j++)
            {
                if (this.array[i + j * res->size_array] < min)
                {
                    min = this.array[i + j * res->size_array];
                    index = j;
                }
            }
            res->array[i] = index;
        }
    }
    else if (axis == 1)
    {
        for (int i = 0; i < res->size_array; i++)
        {
            VAR min = 0;
            int index = 0;
            for (int j = 0; j < this.shape[axis]; j++)
            {
                if (this.array[i * this.shape[axis] + j] < min)
                {
                    min = this.array[i * this.shape[axis] + j];
                    index = j;
                }
            }
            res->array[i] = index;
        }
    }
    return res;    
}


Matrix* __Argmin_Matrix(Matrix this, int axis)
{
    int* shape = malloc(sizeof(int) * this.size_shape-1);
    int k = 0;
    for (int i = 0; i < this.size_shape; i++)
    {
        if (i != axis)
        {
            shape[k] = this.shape[i];
            k++;
        }
    }
    Matrix* res = Init_Matrix(shape, this.size_shape-1, "null");
    if (this.size_shape == 2)
    {
        res = __Argmin_2D(this, axis, res);
    }
    else 
    {
        printf("Not implemented yet");
    }
    free(shape);
    return res;
}
void Argmin_Matrix(Matrix* this, int axis)
{
    Matrix* res = __Argmin_Matrix(*this, axis);
    this->Copy(this, res);
    res->Free(res);
}


