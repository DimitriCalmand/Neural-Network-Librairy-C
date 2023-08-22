#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>

Matrix* __Sum_2D(Matrix this, int axis, Matrix* res)
{
    if (axis == 0)
    {
        for (int i = 0; i < res->size_array; i++)
        {
            for (int j = 0; j < this.shape[axis]; j++)
            {
                res->array[i] += this.array[i + j * res->size_array];
            }
        }
    }
    else if (axis == 1)
    {
        for (int i = 0; i < res->size_array; i++)
        {
            for (int j = 0; j < this.shape[axis]; j++)
            {
                res->array[i] += this.array[i * this.shape[axis] + j];
            }
        }
    }
    return res;    
}
Matrix* __Sum_3D(Matrix this, int axis, Matrix* res)
{
    if (axis == 0)
    {
        int size = this.shape[2]*this.shape[1];
        for (int j = 0; j!= this.shape[1]; j++)
        {
            for (int k = 0; k!= this.shape[2]; k++)
            {
                for (int i = 1; i!= this.shape[0]; i++)
                {
                    VAR tmp = this.array[size*i+j*this.shape[2]+k];
                    res->array[j*res->shape[1]+k] += tmp;
                }
            }            
        }
    }
    else if (axis == 1)
    {
        int size0 = this.shape[1]*this.shape[2];
        for (int i = 0; i!= this.shape[0]; i++)
        {
            for (int j = 0; j!= this.shape[2]; j++)
            {
                for (int k = 0; k!= this.shape[1]; k++)
                {
                    res->array[i*res->shape[1]+j] += this.array[i*size0+k*this.shape[2]+j];
                }
            }
        }
    }
    else 
    {
        int size0 = this.shape[1]*this.shape[2];
        for (int i =0 ; i!= this.shape[0]; i++)
        {
            for (int j = 0; j!= this.shape[1]; j++)
            {
                for (int k = 0; k!= this.shape[2]; k++)
                {
                    res->array[i*res->shape[1]+j] += this.array[i*size0+j*this.shape[2]+k]; 
                }
            }
        }
    }
    return res;
}
void __Sum_Matrix_Place(Matrix this, int axis, Matrix* res)
{
    if (axis < 0)
    {
        axis = this.size_shape + axis;
    }
    if (this.size_shape == 2)
    {
        __Sum_2D(this, axis, res);
    }
    else if (this.size_shape == 3)
    {
        __Sum_3D(this, axis, res);
    }
    else 
    {
        errx(2, "not implemented yet : argmax ");
    }
}
Matrix* __Sum_Matrix(Matrix this, int axis)
{   
    if (axis < 0)
    {
        axis = this.size_shape + axis;
    }
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
    Matrix* res = Init_Matrix(shape, this.size_shape-1, "0");
    __Sum_Matrix_Place(this, axis, res);
    free(shape);
    return res;
}
void Sum_Matrix(Matrix* this, int axis)
{
    Matrix* res = __Sum_Matrix(*this, axis);
    this->Copy(this, res);
    res->Free(res);
}
