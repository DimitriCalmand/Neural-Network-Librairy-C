#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include "Copy.h"
#include <string.h>
#include <time.h>
#include "Copy.h"

Matrix *__Extand_First_Axis(Matrix m1, Matrix m2)
{
    /*
    Return a new matrix with the first axis not equal expanded
    @param
    M1: The matrix
    M2: The matrix to copy
    @return
    The new matrix
    */
    int* shape_m2 ;
    Matrix *m_1 = m1.__Copy(m1);
    if (m_1->size_shape<m2.size_shape)
    {
        for (int i = m_1->size_shape; i!= m2.size_shape; i++)
        {
            m_1->Expand_Dim(m_1, 0);
        }
        shape_m2 = m2.shape;
    }
    else if (m_1->size_shape>m2.size_shape)
    {
        shape_m2 = calloc(m_1->size_shape, sizeof(int));
        int diff = m_1->size_shape - m2.size_shape;
        for (int i = 0; i!= diff; i++)
        {
            shape_m2[i] = 1;
        }
        memcpy(shape_m2+diff, m2.shape, m2.size_shape*sizeof(int));
    }
    else
    {
        shape_m2 = m2.shape;
    }
    int i = 0;
    int end = 0;
    while (i != m_1->size_shape && end == 0)
    {
        if (m_1->shape[i] != shape_m2[i] && m_1->shape[i] == 1)
        {
            end = 1;
            Matrix *res = m_1->__Expand(*m_1, i, shape_m2[i]);
            m_1->Free(m_1);
            if (m_1->size_shape>m2.size_shape)
            {
                free(shape_m2);
            }
            return res;
        }
        i++;
    }
    if (m_1->size_shape>m2.size_shape)
    {
        free(shape_m2);
    }
    return m_1;
}
void __Expand_On_Middle(Matrix m1, Matrix *res, int axis, int nbr_duplication)
{
    // Calule the step for each axis
    int i = m1.size_shape - 1;
    int step = 1;
    while (i != axis)
    {
        step *= m1.shape[i];
        i--;
    }

    VAR *array = res->array;

    for (int j = 0; j != m1.size_array; j += step)
    {
        for (int i = 0; i != nbr_duplication; i++)
        {
            memcpy(array, &m1.array[j], step * sizeof(VAR));
            array += step;
        }
    }
}

Matrix *__Expand_Matrix(Matrix m1, int axis, int nbr_duplication)
{

    if (axis < 0)
    {
        axis = m1.size_shape + axis;
    }
    if (m1.shape[axis] > 1)
    {
        errx(1, "The axis must be of size 1");
    }
    int *shape = malloc(sizeof(int) * (m1.size_shape));
    for (int i = 0; i != m1.size_shape; i++)
    {
        shape[i] = m1.shape[i];
    }
    shape[axis] = nbr_duplication;

    Matrix *res = Init_Matrix(shape, m1.size_shape, "null");

    int *shape1 = malloc(res->size_shape * sizeof(int));
    int *shape2 = malloc(res->size_shape * sizeof(int));

    shape1[0] = m1.shape[m1.size_shape - 1];
    shape2[0] = res->shape[res->size_shape - 1];
    for (int i = 1; i != res->size_shape; i++)
    {
        if (i >= m1.size_shape)
        {
            shape1[i] = shape1[i - 1];
        }
        else
        {
            shape1[i] = m1.shape[m1.size_shape - 1 - i] * shape1[i - 1];
        }
        shape2[i] = res->shape[res->size_shape - i - 1] * shape2[i - 1];
    }
        
    int acc = 0;
    if (axis == 0)
    {
        int size = m1.size_array;
        VAR *array1 = m1.array;
        VAR *array2 = res->array;
        for (int k = 0; k != nbr_duplication; k++)
        {
            memcpy(array2, array1, size * sizeof(VAR));
            array2 += size;
        }
    }
    else if (axis == res->size_shape - 1)
    {
        #if USE_SIMD == 1
        {
            for (int i = 0; i != m1.size_array; i++)
            {
                int k = 0;
                VAR val = m1.array[i];
                for (; k < nbr_duplication - 8; k += 8)
                {
                    __m256 a = _mm256_set_ps(val, val, val, val, val, val, val, val);
                    _mm256_storeu_ps(&res->array[acc], a);
                    acc += 8;
                }
                for (; k < nbr_duplication; k++)
                {
                    res->array[acc] = val;
                    acc++;
                }
            }
        }
        #else
        {
            for (int i = 0; i != m1.size_array; i++)
            {
                VAR val = m1.array[i];
                for (int k = 0; k != nbr_duplication; k++)
                {
                    res->array[acc] = val;
                    acc++;
                }
            }
        }
        #endif
    }
    else
    {
        
        __Expand_On_Middle(m1, res, axis, nbr_duplication);
    }
    free(shape1);
    free(shape2);
    free(shape);
    return res;
}
void Expand_Matrix(Matrix *m1, int axis, int nb_duplication)
{
    Matrix *res = __Expand_Matrix(*m1, axis, nb_duplication);
    m1->Copy(m1, res);
    res->Free(res);
}
void Expand_Dim_Matrix(Matrix *m1, int axis)
{
    m1->shape = realloc(m1->shape, sizeof(int) * (++m1->size_shape));
    for (int i = m1->size_shape - 1; i != axis; i--)
    {
        m1->shape[i] = m1->shape[i - 1];
    }
    m1->shape[axis] = 1;
}
Matrix *__Expand_Dim_Matrix(Matrix m1, int axis)
{
    Matrix *res = m1.__Copy(m1);
    res->Expand_Dim(res, axis);
    return res;
}
