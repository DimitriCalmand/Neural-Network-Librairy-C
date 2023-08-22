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


Matrix* __Mult_Scalar_Matrix(Matrix this, VAR scalar)
{
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    #if USE_SIMD == 2
    {
        cblas_dscal(this.size_array, scalar, this.array, 1);
    }
    #elif USE_SIMD == 1
    {
        // Create a vector of 8 VARs with the scalar value
        __m256 scalar_vec = _mm256_set1_ps(scalar);

        // Add the scalar vector to each 8 elements in the matrix
        for (int i = 0; i < this.size_array-8; i += 8)
        {
            __m256 vec = _mm256_loadu_ps(&this.array[i]);
            __m256 result = _mm256_mul_ps(vec, scalar_vec);
            _mm256_storeu_ps(&res->array[i], result);        
        }
        for (int i = this.size_array/8*8; i!=this.size_array; i++)
        {
            res->array[i] = this.array[i]*scalar;
        }
    }
    #else
    {
        for (int i = 0; i!=this.size_array; i++)
        {
            res->array[i] = this.array[i]*scalar;
        }
    }
    #endif
    return res;
}

void Mult_Scalar_Matrix(Matrix* this, VAR scalar)
{
    /*
    Add a scalar to each element of the matrix
    @param
    THIS: The matrix
    SCALAR: The scalar to add
    */
   #if USE_SIMD==2
    {
        cblas_dscal(this->size_array, scalar, this->array, 1);
    }
    #elif USE_SIMD==1
    {
        VAR* array = this->array;
        __m256 scalar_vec = _mm256_set1_ps(scalar);

        for (int i = 0; i < this->size_array-8; i += 8)
        {
            __m256 a = _mm256_loadu_ps(array);
            __m256 b = _mm256_mul_ps(a, scalar_vec);
            _mm256_storeu_ps(array, b);
            array += 8;
        }
        for (int i = this->size_array/8*8; i!=this->size_array; i++)
        {
            this->array[i]*=scalar;
        }
    }
    #else
    {
        for (int i = 0; i < this->size_array; i++)
        {
            this->array[i]*=scalar;
        }   
    }
    #endif
}
void __Mult_Mat(Matrix* m1, Matrix* m2, Matrix* res)
{
    #if USE_SIMD==2
    {
        vdMul(m1->size_array, m1->array, m2->array, res->array);
    }
    #elif USE_SIMD==1
    {
        for (int i = 0; i<m1->size_array-8; i+=8)
        {
            __m256 v1 = _mm256_loadu_ps(m1->array + i);
            __m256 v2 = _mm256_loadu_ps(m2->array + i);
            _mm256_storeu_ps(res->array + i, _mm256_mul_ps(v1, v2));
        }
        for (int i = m1->size_array/8*8; i!=m1->size_array; i++)
        {
            res->array[i]=m2->array[i]*m1->array[i];
        }
    }
    #else
    {
        for (int i = 0; i<m1->size_array; i++)
        {
            res->array[i]=m2->array[i]*m1->array[i];
        }
    }
    #endif
}
Matrix* __Mult_Matrix_Bad_Shape(Matrix m_1, Matrix m_2, Matrix* res)
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
            "the matrix doesn't have the correct dimensions for multiplication in __Mult_Matrix");
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
        __Mult_Mat(m1, m2, m1);
        m2->Free(m2);
        return m1;
    }
    __Mult_Mat(m1, m2, res);
    m2->Free(m2);
    m1->Free(m1);
    return NULL;
    
}
void __Mult_Mat_Place(Matrix m_1, Matrix m_2, Matrix* res)
{
    if (m_1.size_shape != m_2.size_shape)
    {
        __Mult_Matrix_Bad_Shape(m_1, m_2, res);
        return;
    }
    for (int i = 0; i!= m_1.size_shape; i++)
    {
        if (m_1.shape[i] != m_2.shape[i])
        {
            __Mult_Matrix_Bad_Shape(m_1, m_2, res);
            return;
        }
    }
    __Mult_Mat(&m_1, &m_2, res);
}
Matrix* __Mult_Matrix(Matrix m_1, Matrix m_2)
{
    if (m_1.size_shape != m_2.size_shape)
    {
        return __Mult_Matrix_Bad_Shape(m_1, m_2, NULL);
    }
    for (int i = 0; i!= m_1.size_shape; i++)
    {
        if (m_1.shape[i] != m_2.shape[i])
        {
            return __Mult_Matrix_Bad_Shape(m_1, m_2, NULL);
        }
    }
    Matrix* res = Init_Matrix(m_1.shape, m_1.size_shape, "null");
    __Mult_Mat(&m_1, &m_2, res);
    return res;

}
void Mult_Matrix(Matrix* m1, Matrix* m_2)
{
    /*
    Add the two matrix
    @param
    M1: The first matrix
    M2: The second matrix
    */
    __Mult_Mat_Place(*m1, *m_2, m1);
}

Matrix* __Dot_Matrix_3D_2D(Matrix m1, Matrix m2)
{
    /*
    Return a new matrix with the dot product of the two matrix (only for 3D matrix)
    @param
    M1: The first matrix (3D)
    M2: The second matrix (2D)
    @return
    The new matrix
    */
    // Make sure the shapes are compatible for dot product
    if (m1.size_shape != 3 || m2.size_shape != 2 || m1.shape[2] != m2.shape[0])
    {
        errx(1, "The shapes are not compatible for dot product");
    }
    // Initialize the result matrix
    int shape[] = {m1.shape[0], m1.shape[1], m2.shape[1]};
    Matrix* res = Init_Matrix(shape, 3, "null");
    #if USE_SIMD == 2
    {
        for (int i = 0; i < m1.shape[0]; i++)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.shape[1], m2.shape[1], m1.shape[2],
            1.0, m1.array+i * m1.shape[1] * m1.shape[2], m1.shape[2], m2.array, m2.shape[1], 0.0, res->array+i * res->shape[1] * res->shape[2], m2.shape[1]);
        }
    }
    #else
    {
        int size_m1_1_2 = m1.shape[1] * m1.shape[2];
        int size_res_1_2 = res->shape[1] * res->shape[2];
        int index_1_m1 = 0;
        int index_2_m1 = 0;
        int index_res_1 = 0;
        int index_res_2 = 0;
        for (int i = 0; i < res->shape[0]; i++)
        {
            index_1_m1 = i * size_m1_1_2;
            index_res_1 = i * size_res_1_2;
            for (int j = 0; j < res->shape[1]; j++)
            {
                index_2_m1 = index_1_m1 + j * m1.shape[2];
                index_res_2 = index_res_1 + j * res->shape[2];
                for (int k = 0; k < res->shape[2]; k++)
                {
                    VAR dot_product = 0;

                    for (int l = 0; l < m1.shape[2]; l++)
                    {
                        dot_product += m1.array[index_2_m1 + l] * m2.array[l * m2.shape[1] + k];
                    }
                    res->array[index_res_2 + k] = dot_product;
                }
            }
        }
    }
    #endif
    return res;
}
    void __Dot_Matrix_2D_Place(Matrix m1, Matrix m2, Matrix* res)
{
    #if USE_SIMD==2
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.shape[0], m2.shape[1], m1.shape[1], 1.0, m1.array, m1.shape[1], m2.array, m2.shape[1], 0.0, res->array, m2.shape[1]);
    }
    #elif USE_SIMD==1
    {
        VAR* array1 = m1.array;
        int row_m1 = m1.shape[0];
        int col_m2 = m2.shape[1];
        int col_m1 = m1.shape[1];
        int index_m1 = 0;
        int index_res = 0;
        int i, j, k;
        Matrix* tmp = m2.__Transpose(m2,0,1);
        VAR* array2 = tmp->array;
        
        for (i = 0; i < row_m1; ++i) 
        {
            index_m1 = i*col_m1;    
            index_res = i*col_m2;
            for (j = 0; j < col_m2; ++j) 
            {
                res->array[index_res+j] = 0;
                k = 0;
                for (; k < col_m1-8; k+=8)
                {
                    __m256 v1 = _mm256_loadu_ps(array1 + index_m1 + k);
                    __m256 v2 = _mm256_loadu_ps(array2 + j*col_m1+k);
                    v2 = _mm256_mul_ps(v1, v2);
                    res->array[index_res + j] += v2[0] + v2[1] + v2[2] + v2[3] + v2[4] + v2[5] + v2[6] + v2[7];
                }
                for (; k < col_m1; k++)
                {
                    res->array[index_res + j] += array2[j*col_m1+k] * array1[i*col_m1+k];
                }
            }
        }
        tmp->Free(tmp);
    }
    #else
    {
        int row_m1 = m1.shape[0];
        int col_m2 = m2.shape[1];
        int col_m1 = m1.shape[1];
        VAR* array1 = m1.array;
        int index_m1 = 0;
        int index_res = 0;
        VAR* array2 = m2.array;
        for (int i = 0; i < row_m1; ++i) 
        {
            index_m1 = i*col_m1;
            index_res = i*col_m2;
            for (int j = 0; j < col_m2; ++j) 
            {
                res->array[index_res+j] = 0;
                for (int k = 0; k < col_m1; ++k)
                {
                    VAR v1 = array1[index_m1+k];
                    VAR v2 = array2[k*col_m2+j];
                    res->array[index_res+j] += v1*v2;
                }
            }
        }
    }
    #endif
}
Matrix* __Dot_Matrix_2D_2D(Matrix m1, Matrix m2)
{
    /*
    Return a new matrix with the dot product of the two matrix (only for 2D matrix)
    @param
    M1: The first matrix (2D)
    M2: The second matrix (2D)
    @return
    The new matrix
    */
    if (m1.size_shape!=2 || m2.size_shape!=2)
    {
        errx(EXIT_FAILURE, "The matrix doesn't have the correct dimensions to dot");
    }
    if (m1.shape[1]!=m2.shape[0])
    {
        errx(EXIT_FAILURE, "The matrix doesn't have the correct dimensions to dot find m1 shape[1] = %d, m2 shape[0] = %d", m1.shape[1], m2.shape[0]);
    }
    int row_m1 = m1.shape[0];
    int col_m2 = m2.shape[1];
    int shape[] = {row_m1, col_m2};    
    Matrix* res = Init_Matrix(shape, 2, "null");
    __Dot_Matrix_2D_Place(m1, m2, res);
    return res;
}
Matrix* __Dot_Matrix(Matrix m1, Matrix m2)
{
    /*
    Return a new matrix with the dot product of the two matrix
    @param
    M1: The first matrix
    M2: The second matrix
    @return
    The new matrix
    */
   //printf("m1 size shape = %d, m2 size shape = %d\n", m1.size_shape, m2.size_shape);
    if (m1.size_shape==3 && m2.size_shape==2)
    {
        return __Dot_Matrix_3D_2D(m1, m2);
    }
    else if (m1.size_shape==2 && m2.size_shape==2)
    {
        return __Dot_Matrix_2D_2D(m1, m2);
    }
    else
    {
        errx(EXIT_FAILURE, "The matrix doesn't have the correct dimensions to dot find m1 size shape = %d, m2 size shape = %d", m1.size_shape, m2.size_shape);
    }
}
void Dot_Matrix(Matrix* m1, Matrix* m2)
{
    /*
    Dot two matrix (only for 3D and 2D matrix)
    @param
    M1: The first matrix (3D)
    M2: The second matrix (2D)
    */
    Matrix* res = __Dot_Matrix(*m1, *m2);
    Copy_Matrix(m1, res);
    res->Free(res);
}

