#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Math_Utils.h"
#include "../Matrix.h"
#include "../utils/Copy.h"
#include <string.h>
#include <time.h>

void __Power(Matrix* this, VAR power, Matrix* res)
{
    #if USE_SIMD == 2
    {
        vdPowx(this->size_array, this->array, power, res->array); 
    }
    #else
    {
        for (int i = 0; i!=this->size_array; i++)
        {   
            res->array[i] = pow(this->array[i], power);
        }
    }
    #endif
}
Matrix* __Power_Matrix(Matrix this, VAR power)
{
    /*
    Return a new matrix with the power of each element of the matrix
    @param
    THIS: The matrix
    @return
    The new matrix
    */
    Matrix* res = Init_Matrix(this.shape, this.size_shape, "null");
    __Power(&this, power, res);
    return res;
}
void Power_Matrix(Matrix *this, VAR power) {
    /*
    Apply the power to each element of the matrix in-place
    @param
    THIS: The matrix
    */

    #if USE_SIMD == 2
    {
        vdPowx(this->size_array, this->array, power, this->array);
    }
    #else
    {
        for (int i = 0; i != this->size_array; i++) {
            this->array[i] = pow(this->array[i], power);
        }
    }
    #endif
}
