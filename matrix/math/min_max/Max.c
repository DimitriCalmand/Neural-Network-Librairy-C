#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../../Matrix.h"
#include <string.h>
#include <time.h>

VAR Max_Matrix(Matrix* this)
{
    VAR max = 0;
    for (int i = 0; i < this->size_array; i++)
    {
        if (this->array[i] > max)
        {
            max = this->array[i];
        }
    }
    return max;
}

Matrix* __Argmax_2D(Matrix this, int axis, Matrix* res)
{
    if (axis == 0)
    {
        for (int i = 0; i < res->size_array; i++)
        {
            VAR max = 0;
            int index = 0;
            for (int j = 0; j < this.shape[axis]; j++)
            {
                if (this.array[i + j * res->size_array] > max)
                {
                    max = this.array[i + j * res->size_array];
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
            VAR max = 0;
            int index = 0;
            for (int j = 0; j < this.shape[axis]; j++)
            {
                if (this.array[i * this.shape[axis] + j] > max)
                {
                    max = this.array[i * this.shape[axis] + j];
                    index = j;
                }
            }
            res->array[i] = index;
        }
    }
    return res;    
}
Matrix* __Argmax_3D(Matrix this, int axis, Matrix* res)
{
    if (axis == 0)
    {
        int size = this.shape[2]*this.shape[1];
        for (int j = 0; j!= this.shape[1]; j++)
        {
            for (int k = 0; k!= this.shape[2]; k++)
            {
                int index = 0;
                VAR maxi = this.array[j*this.shape[2]+k];
                for (int i = 1; i!= this.shape[0]; i++)
                {
                    VAR tmp = this.array[size*i+j*this.shape[2]+k];
                    if (maxi < tmp)
                    {
                        maxi = tmp;
                        index = i;
                    }
                }
                res->array[j*res->shape[1]+k] = index;
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
                VAR maxi = this.array[i*size0+j];
                int index = 0;

                for (int k = 0; k!= this.shape[1]; k++)
                {
                    if (maxi<this.array[i*size0+k*this.shape[2]+j])
                    {
                        maxi = this.array[i*size0+k*this.shape[2]+j];
                        index = k;
                    }
                }
                res->array[i*res->shape[1]+j] = index;
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
                VAR maxi = this.array[i*size0+j*this.shape[2]];
                int index = 0;
                for (int k = 0; k!= this.shape[2]; k++)
                {
                    if (this.array[i*size0+j*this.shape[2]+k]> maxi)
                    {
                        maxi = this.array[i*size0+j*this.shape[2]+k];
                        index = k;
                    }
                }
                res->array[i*res->shape[1]+j] = index;
            }
        }
    }
    return res;
}
Matrix* __Argmax_Matrix(Matrix this, int axis)
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
    Matrix* res = Init_Matrix(shape, this.size_shape-1, "1000");
    if (this.size_shape == 2)
    {
        __Argmax_2D(this, axis, res);
    }
    else if (this.size_shape == 3)
    {
        __Argmax_3D(this, axis, res);
    }
    else 
    {
        errx(2, "not implemented yet : argmax ");
    }
    free(shape);
    return res;
}
void Argmax_Matrix(Matrix* this, int axis)
{
    Matrix* res = __Argmax_Matrix(*this, axis);
    this->Copy(this, res);
    res->Free(res);
}


