#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>
#include "Math_Utils.h"
#include "../utils/Expand.h"
#include <time.h>



Matrix* __Add_Scalar_Matrix(Matrix this, VAR scalar)
{
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");

    #if USE_SIMD == 2
    {
        for (int i = 0; i!=this.size_array; i++)
        {
            res->array[i] = this.array[i]+scalar;
        }
    }
    #elif USE_SIMD == 1
    {
        __m256 scalar_vec = _mm256_set1_ps(scalar);

        // Add the scalar vector to each 8 elements in the matrix
        for (int i = 0; i < this.size_array-8; i += 8)
        {
            __m256 vec = _mm256_loadu_ps(&this.array[i]);
            __m256 result = _mm256_add_ps(vec, scalar_vec);
            _mm256_storeu_ps(&res->array[i], result);
        }
        for (int i = this.size_array/8*8; i!=this.size_array; i++)
        {
            res->array[i] = this.array[i]+scalar;
        }
    }
    #else
    {
        for (int i = 0; i!=this.size_array; i++)
        {
            res->array[i] = this.array[i]+scalar;
        }
    }
    #endif
    return res;
}

void Add_Scalar_Matrix(Matrix* this, VAR scalar)
{
    /*
    Add a scalar to each element of the matrix
    @param
    THIS: The matrix
    SCALAR: The scalar to add
    */
    #if USE_SIMD == 2
    {
        for (int i = 0; i!=this->size_array; i++)
        {
            this->array[i]+=scalar;
        }
    }
    #elif USE_SIMD == 1
    {
        VAR* array = this->array;
        __m256 scalar_vec = _mm256_set1_ps(scalar);
        for (int i = 0; i < this->size_array-8; i += 8)
        {
            __m256 a = _mm256_loadu_ps(array);
            __m256 b = _mm256_add_ps(a, scalar_vec);
            _mm256_storeu_ps(array, b);
            array += 8;
        }
        for (int i = this->size_array/8*8; i!=this->size_array; i++)
        {
            this->array[i]+=scalar;
        }
    }
    #else
    {
        for (int i = 0; i!=this->size_array; i++)
        {
            this->array[i]+=scalar;
        }
    }
    #endif

}

void __Add_Mat(Matrix* m1, Matrix* m2, Matrix* res)
{
    // printf("Enter\n");
    // m2->Print(m2);
    // m1->Print(m1);
    // res->Print(res);
    // printf("Close\n");
    #if USE_SIMD == 2
    {
        vdAdd(m1->size_array, m1->array, m2->array, res->array);
        // cblas_daxpy(m1->size_array, 1.0, m2->array, 1, m1->array, 1);
    }
    #elif USE_SIMD == 1
    {
        for (int i = 0; i<m1->size_array-8; i+=8)
        {
            __m256 v1 = _mm256_loadu_ps(m1->array + i);
            __m256 v2 = _mm256_loadu_ps(m2->array + i);
            _mm256_storeu_ps(res->array + i, _mm256_add_ps(v1, v2));
        }
        for (int i = m1->size_array/8*8; i!=m1->size_array; i++)
        {
            res->array[i] = m1->array[i]+m2->array[i];
        }
    }
    #else
    {
        for (int i = 0; i!=m1->size_array; i++)
        {
            res->array[i] = m1->array[i]+m2->array[i];
        }
    }  
    #endif
}
Matrix* __Add_Matrix_Bad_Shape(Matrix m_1, Matrix m_2, Matrix* res)
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
            "the matrix doesn't have the correct dimensions for addition.c");
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
    if (res == NULL)
    {
        __Add_Mat(m1, m2, m1);
        m2->Free(m2);
        return m1;
    }
    __Add_Mat(m1, m2, res);
    m2->Free(m2);
    m1->Free(m1);
    return NULL;
}
void __Add_Mat_Place(Matrix m_1, Matrix m_2, Matrix* res)
{
    
    if (m_1.size_shape != m_2.size_shape)
    {
        // printf("henter\n");
        // m_1.Print_Shape(&m_1);
        // m_2.Print_Shape(&m_2);
        __Add_Matrix_Bad_Shape(m_1, m_2, res);
        return;
    }
    for (int i = 0; i!= m_1.size_shape; i++)
    {
        if (m_1.shape[i] != m_2.shape[i])
        {
            __Add_Matrix_Bad_Shape(m_1, m_2, res);
            return;
        }
    }
    __Add_Mat(&m_1, &m_2, res);
}
Matrix* __Add_Matrix(Matrix m_1, Matrix m_2)
{
    if (m_1.size_shape != m_2.size_shape)
    {
        return __Add_Matrix_Bad_Shape(m_1, m_2, NULL);
    }
    for (int i = 0; i!= m_1.size_shape; i++)
    {
        if (m_1.shape[i] != m_2.shape[i])
        {
            return __Add_Matrix_Bad_Shape(m_1, m_2, NULL);
        }
    }
    Matrix* res = Init_Matrix(m_1.shape, m_1.size_shape, "null");
    __Add_Mat(&m_1, &m_2, res);
    return res;
}

void Add_Matrix(Matrix* m1, Matrix* m_2)
{
    /*
    Add the two matrix
    @param
    M1: The first matrix
    M2: The second matrix
    */
    __Add_Mat_Place(*m1, *m_2, m1);
}

