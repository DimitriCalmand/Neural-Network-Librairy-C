#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../utils/Copy.h"
#include "Math_Utils.h"
#include "../Matrix.h"
#include <string.h>
#include <time.h>


void __Transpose_Matrix_2d(Matrix this, Matrix* res)
{
    for (int i = 0; i!= res->shape[0]; i++)
    {
        for (int j = 0; j!= res->shape[1]; j++)
        {
            res->array[i*res->shape[1]+ j] = 
                this.array[i+j*res->shape[0]];
        }
    }
}
void __Transpose_Matrix_3d(Matrix this, int p1, int p2, Matrix* res)
{
    if ((p1==0 && p2==1) || (p1==1 && p2==0))
    {
        for (int i = 0; i!= res->shape[0]; i++)
        {
            for (int j = 0; j!= res->shape[1]; j++)
            {
                for (int k = 0; k!= res->shape[2]; k++)
                {
                    res->array[i*res->shape[1]*res->shape[2]+ j*res->shape[2]+ k] =
                    this.array[j*this.shape[1]*this.shape[2] + i*this.shape[2] + k];
                }
            }
        }
    }
    else if ((p1==1 && p2==2) || (p1==2 && p2==1))
    {
        for (int i = 0; i!= res->shape[0]; i++)
        {
            for (int j = 0; j!= res->shape[1]; j++)
            {
                for (int k = 0; k!= res->shape[2]; k++)
                {
                    res->array[i*res->shape[1]*res->shape[2]+ j*res->shape[2]+ k] =
                    this.array[i*this.shape[1]*this.shape[2] + k*this.shape[2] + j];
                }
            }
        }
    }
    else
    {
        printf("p1: %d, p2: %d in transpose.c \n", p1, p2);
        for (int i = 0; i!= res->shape[0]; i++)
        {
            for (int j = 0; j!= res->shape[1]; j++)
            {
                for (int k = 0; k!= res->shape[2]; k++)
                {
                    res->array[i*res->shape[1]*res->shape[2]+ j*res->shape[2]+ k] =
                    this.array[k*this.shape[1]*this.shape[2] + j*this.shape[2] + i];
                }
            }
        }
    }
}
void __Transpose_Matrix_Place(Matrix this, int p1, int p2, Matrix* res)
{
    if (this.size_shape<=1)
    {
        errx(EXIT_FAILURE, "The matrix doesn't have the correct dimensions to tranpose");
    }
    if (p1<0)
    {
        p1 = this.size_shape + p1;
    }
    if (p2<0)
    {
        p2 = this.size_shape + p2;
    }
    int sum_size = 1;
    for (int i = 0; i!=this.size_shape; i++)
    {
        sum_size *= this.shape[i];
    }
    if (this.size_shape==2)
    {
        __Transpose_Matrix_2d(this, res);
    }
    else if (this.size_shape==3)
    {
        __Transpose_Matrix_3d(this, p1, p2, res);
    }
    else
    {
        errx(EXIT_FAILURE, "The transpose function is not implemented for this dimension: Transpose.c");
    }
}
Matrix* __Transpose_Matrix(Matrix this, int p1, int p2)
{
    /*
    Return a new matrix with the transpose of the matrix    
    @param
    THIS: The matrix
    P1: The first position
    P2: The second position
    @return
    The new matrix
    */
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    if (this.size_shape<=1)
    {
        errx(EXIT_FAILURE, "The matrix doesn't have the correct dimensions to tranpose");
    }
    if (p1<0)
    {
        p1 = this.size_shape + p1;
    }
    if (p2<0)
    {
        p2 = this.size_shape + p2;
    }
    int tmp = res->shape[p1];
    res->shape[p1] = res->shape[p2];
    res->shape[p2] = tmp;
    __Transpose_Matrix_Place(this, p1, p2, res);    
    return res ;
}
void Transpose_Matrix(Matrix* this, int p1, int p2)
{
    /*
    Transpose the matrix
    @param
    THIS: The matrix
    P1: The first position
    P2: The second position
    */
    Matrix* tmp_transpose = __Transpose_Matrix(*this, p1, p2);
    Copy_Matrix(this,tmp_transpose);
    tmp_transpose->Free(tmp_transpose);
}