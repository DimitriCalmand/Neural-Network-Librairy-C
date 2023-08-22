#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../utils/Copy.h"
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include <string.h>
#include <time.h>
#include "Math_Utils.h"
#include "../utils/Expand.h"

Matrix* __Sub_Matrix(Matrix m_1, Matrix m_2)
{
    Matrix* m1 = __Extand_First_Axis(m_1, m_2);
    Matrix* m2 = __Extand_First_Axis(m_2, m_1);
    for (int i = 0; i!=m1->size_shape; i++)
    {
        int v1 = m1->shape[m1->size_shape-1-i];
        int v2 = m2->shape[m2->size_shape-1-i];
        if (v1 != v2 && !(v1==1 || v2==1))
        {
            errx(EXIT_FAILURE, 
            "the matrix doesn't have the correct dimensions for Substraction in __Sub_Matrix");
        }
        //Expand the shape of the minimum matrix
        if (v1>v2)
        {
            if (v2==1)
            {
                m2->Expand(m2, -1-i, v1);
            }
        }
        else if (v1!=v2)
        {
            if (v1==1)
            {
                m1->Expand(m1, -1-i, v2);
            }
        }        
    }
    #if USE_SIMD == 2
    {
        cblas_daxpy(m1->size_array, -1.0, m2->array, 1, m1->array, 1);
    }
    #elif USE_SIMD == 1
    {
        for (int i = 0; i<m1->size_array-8; i+=8)
        {
            __m256 v1 = _mm256_loadu_ps(m1->array + i);
            __m256 v2 = _mm256_loadu_ps(m2->array + i);
            _mm256_storeu_ps(m1->array + i, _mm256_sub_ps(v1, v2));
        }
        for (int i = m1->size_array/8*8; i!=m1->size_array; i++)
        {
            m1->array[i]-=m2->array[i];
        }
    }
    #else
    {
        for (int i = 0; i!=m1->size_array; i++)
        {
            m1->array[i]-=m2->array[i];
        }
    }
    #endif
    m2->Free(m2);
    return m1;

}
void Sub_Matrix(Matrix* m1, Matrix* m_2)
{
    /*
    Add the two matrix
    @param
    M1: The first matrix
    M2: The second matrix
    */
    Matrix* m2 = __Extand_First_Axis(*m_2, *m1);
    for (int i = 0; i!=m1->size_shape; i++)
    {
        int v1 = m1->shape[m1->size_shape-1-i];
        int v2 = m2->shape[m2->size_shape-1-i];
        if (v1 != v2 && !(v1==1 || v2==1))
        {
            errx(EXIT_FAILURE, 
            "the matrix doesn't have the correct dimensions for substraction in Sub_Matrix");
        }
        //Expand the shape of the minimum matrix
        if (v1>v2)
        {
            if (v2==1)
            {
                m2->Expand(m2, -1-i, v1);

            }           
        }
        else if (v1!=v2)
        {
            if (v1==1)
            {
                m1->Expand(m1, -1-i, v2);
            }
        }    
    }
    #if USE_SIMD == 2
    {
        cblas_daxpy(m1->size_array, -1.0, m2->array, 1, m1->array, 1);
    }
    #elif USE_SIMD == 1
    {
        for (int i = 0; i<m1->size_array-8; i+=8)
        {
            __m256 v1 = _mm256_loadu_ps(m1->array + i);
            __m256 v2 = _mm256_loadu_ps(m2->array + i);
            _mm256_storeu_ps(m1->array + i, _mm256_sub_ps(v1, v2));
        }
        for (int i = m1->size_array/8*8; i!=m1->size_array; i++)
        {
            m1->array[i]-=m2->array[i];
        }
    }
    #else 
    {
        for (int i = 0; i!=m1->size_array; i++)
        {
            m1->array[i]-=m2->array[i];
        }
    }
    #endif
    m2->Free(m2);
}
