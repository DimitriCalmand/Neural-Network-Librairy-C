#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Copy.h"
#include "../Matrix.h"
#include <string.h>
#include <time.h>

int Index_For_Slicing(Matrix this, int index, int axis)
{
    /*
    Return the index of the new matrix
    @param
    THIS: The old matrix
    INDEX: The index of the old matrix
    AXIS: The index of the shape
    @return
    The index of the new matrix
    */
    for (int i = this.size_shape - 1; i != axis; i--)
    {
        index = index / this.shape[i];
    }
    return index % this.shape[axis];
}
void __SlicingNd_Matrix(Matrix this, int axis, int start, int end, Matrix *res)
{
    /*
    Slice the matrix and return a new matrix with the new shape
    @param
    THIS: The matrix
    AXIS: The index of the shape
    START: The start of the slice
    END: The end of the slice
    @return
    The new matrix
    */

    int k = 0;
    for (int i = 0; i != this.size_array; i++)
    {
        int index = Index_For_Slicing(this, i, axis);
        if (index >= start && index < end)
        {
            res->array[k] = this.array[i];
            k++;
        }
    }
}

void __Slincing2d_Matrix(Matrix this, int axis, int start, int end, Matrix *res)
{
    /*
    Slice the matrix and return a new matrix with the new shape
    @param
    THIS: The matrix
    START: The start of the slice
    END: The end of the slice
    @return
    The new matrix
    */
    if (axis == 0)
    {
        for (int i = start; i != end; i++)
        {
            // use memcpy to copy the array
            memcpy(res->array + (i - start) * this.shape[1], this.array + i * this.shape[1], this.shape[1] * sizeof(VAR));
        }
    }
    else
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            memcpy(res->array + i * (end - start), this.array + i * this.shape[1] + start, (end - start) * sizeof(VAR));
        }
    }
}

void __Slicing3d_Matrix(Matrix this, int axis, int start, int end, Matrix *res)
{
    /*
    Slice the matrix and return a new matrix with the new shape
    @param
    THIS: The matrix
    AXIS: The index of the shape
    START: The start of the slice
    END: The end of the slice
    @return
    The new matrix
    */
    if (axis == 0)
    {
        for (int i = start; i != end; i++)
        {
            memcpy(res->array + (i - start) * this.shape[1] * this.shape[2], this.array + i * this.shape[1] * this.shape[2], this.shape[1] * this.shape[2] * sizeof(VAR));
        }
    }
    else if (axis == 1)
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            for (int j = start; j != end; j++)
            {
                memcpy(res->array + i * (end - start) * this.shape[2] + (j - start) * this.shape[2], this.array + i * this.shape[1] * this.shape[2] + j * this.shape[2], this.shape[2] * sizeof(VAR));
            }
        }
    }
    else
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            for (int j = 0; j != this.shape[1]; j++)
            {
                memcpy(res->array + i * this.shape[1] * (end - start) + j * (end - start), this.array + i * this.shape[1] * this.shape[2] + j * this.shape[2] + start, (end - start) * sizeof(VAR));
            }
        }
    }
}

void __Slicing4d_Matrix(Matrix this, int axis, int start, int end, Matrix *res)
{
    /*
    Slice the matrix and return a new matrix with the new shape
    @param
    THIS: The matrix
    AXIS: The index of the shape
    START: The start of the slice
    END: The end of the slice
    @return
    The new matrix
    */
    if (axis == 0)
    {
        for (int i = start; i != end; i++)
        {
            memcpy(res->array + (i - start) * this.shape[1] * this.shape[2] * this.shape[3], this.array + i * this.shape[1] * this.shape[2] * this.shape[3], this.shape[1] * this.shape[2] * this.shape[3] * sizeof(VAR));
        }
    }
    else if (axis == 1)
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            for (int j = start; j != end; j++)
            {
                memcpy(res->array + i * (end - start) * this.shape[2] * this.shape[3] + (j - start) * this.shape[2] * this.shape[3], this.array + i * this.shape[1] * this.shape[2] * this.shape[3] + j * this.shape[2] * this.shape[3], this.shape[2] * this.shape[3] * sizeof(VAR));
            }
        }
    }
    else if (axis == 2)
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            for (int j = 0; j != this.shape[1]; j++)
            {
                for (int k = start; k != end; k++)
                {
                    memcpy(res->array + i * this.shape[1] * (end - start) * this.shape[3] + j * (end - start) * this.shape[3] + (k - start) * this.shape[3], this.array + i * this.shape[1] * this.shape[2] * this.shape[3] + j * this.shape[2] * this.shape[3] + k * this.shape[3], this.shape[3] * sizeof(VAR));
                }
            }
        }
    }
    else
    {
        for (int i = 0; i != this.shape[0]; i++)
        {
            for (int j = 0; j != this.shape[1]; j++)
            {
                for (int k = 0; k != this.shape[2]; k++)
                {
                    memcpy(res->array + i * this.shape[1] * this.shape[2] * (end - start) + j * this.shape[2] * (end - start) + k * (end - start), this.array + i * this.shape[1] * this.shape[2] * this.shape[3] + j * this.shape[2] * this.shape[3] + k * this.shape[3] + start, (end - start) * sizeof(VAR));
                }
            }
        }
    }
}

Matrix *__Slicing_Matrix(Matrix this, int axis, int start, int end)
{
    /*
    Slice the matrix and return a new matrix with the new shape
    @param
    THIS: The matrix
    AXIS: The index of the shape
    START: The start of the slice
    END: The end of the slice
    @return
    The new matrix
    */
    if (axis >= this.size_shape)
    {
        errx(1, "The axis is out of range");
    }
    if (start < 0)
    {
        start = this.shape[axis] + start;
    }
    if (end < 0)
    {
        end = this.shape[axis] + end;
    }
    if (start > end)
    {
        printf("start = %d, end = %d", start, end);
        errx(1, "The start is bigger than the end");
    }
    if (start < 0 || end > this.shape[axis])
    {
        printf("start = %d, end = %d\n", start, end);
        this.Print_Shape(&this);
        errx(1, "The start or the end is out of range : Slicing");
    }
    int size_shape = this.size_shape;
    if (start + 1 == end)
    {
        size_shape--;
    }
    int *shape = malloc((size_shape) * sizeof(int));
    int k = 0;
    for (int i = 0; i != this.size_shape; i++)
    {
        if (i != axis)
        {
            shape[k] = this.shape[i];
            k++;
        }
        else if (start + 1 != end)
        {
            shape[k] = end - start;
            k++;
        }
    }
    Matrix *res = Init_Matrix(shape, size_shape, "null");
    if (this.size_shape == 2)
    {
        __Slincing2d_Matrix(this, axis, start, end, res);
    }
    else if (this.size_shape == 3)
    {
        __Slicing3d_Matrix(this, axis, start, end, res);
    }
    else if (this.size_shape == 4)
    {
        __Slicing4d_Matrix(this, axis, start, end, res);
    }
    else
    {
        __SlicingNd_Matrix(this, axis, start, end, res);
    }
    free(shape);
    return res;
}
void Slicing_Matrix(Matrix *this, int axis, int start, int end)
{
    /*
    Slcing the matrix
    @param
    THIS: The matrix
    AXIS: The index of the shape
    START: The start of the slice
    END: The end of the slice
    */
    Matrix *tmp = __Slicing_Matrix(*this, axis, start, end);
    Copy_Matrix(this, tmp);
    tmp->Free(tmp);
}

void Put_2d(Matrix *this, Matrix *m2, int axis, int start, int end)

{
    VAR *array1 = this->array;
    VAR *array2 = m2->array;

    if (axis == 0)
    {
        memcpy(array1+start*(this->shape[1]), array2, sizeof(VAR)*(end-start)*this->shape[1]);
    } 
    else
    {
        for (int i = 0; i!=this->shape[0]; i++)
        {
            memcpy(array1+i*this->shape[1]+start, array2+i*m2->shape[1], sizeof(VAR)*(end-start));
        }
    }
}
void Put_3d(Matrix* this, Matrix* m2, int axis, int start, int end)
{
    VAR* array1 = this->array;
    
    Matrix* m_2 ;
    if (this->size_shape != m2->size_shape)
    {
        m_2 = m2->__Expand_Dim(*m2, axis);
    }
    else
    {
        m_2 = m2;
    }
    VAR* array2 = m_2->array;
    if (axis == 0)
    {
        memcpy(array1+start*(this->shape[1]*this->shape[2]), array2, sizeof(VAR)*(end-start)*this->shape[1]*this->shape[2]);
    } 
    else if (axis == 1)
    {
        for (int i = 0; i!=this->shape[0]; i++)
        {
            memcpy(array1+i*this->shape[1]*this->shape[2]+start*this->shape[2], array2+i*m_2->shape[1]*m_2->shape[2], sizeof(VAR)*m_2->shape[1]*m_2->shape[2]);
        }
    }
    else
    {
        for (int i = 0; i!=this->shape[0]; i++)
        {
            for (int j = 0; j!=this->shape[1]; j++)
            {
                memcpy(array1+i*this->shape[1]*this->shape[2]+j*this->shape[2]+start, array2+i*m_2->shape[1]*m_2->shape[2]+j*m_2->shape[2], sizeof(VAR)*(end-start));
            }
        }
    }
    
}
void Put_Matrix(Matrix *this, Matrix *m2, int axis, int start, int end)
{
    if (axis >= this->size_shape)
    {
        errx(1, "The axis is out of range");
    }
    if (start < 0)
    {
        start = this->shape[axis] + start;
    }
    if (end < 0)
    {
        end = this->shape[axis] + end;
    }
    if (start > end)
    {
        printf("start = %d, end = %d\n", start, end);
        errx(1, "The start is bigger than the end : Put");
    }
    if (start < 0 || end > this->shape[axis])
    {
        printf("start = %d, end = %d\n", start, end);
        this->Print_Shape(this);
        errx(1, "The start or the end is out of range : Put");
    }
    Matrix* m_2 ;
    if (this->size_shape != m2->size_shape)
    {
        m_2 = m2->__Expand_Dim(*m2, axis);
    }
    else
    {
        m_2 = m2;
    }
    if (this->size_shape == 2)
    {
        Put_2d(this, m_2, axis, start, end);
    }
    if (this->size_shape == 3){
        Put_3d(this, m_2, axis, start, end);
    }
    m_2->Free(m_2);
}