#ifndef MATRIX1_H
#define MATRIX1_H

#include <pthread.h>
#if USE_SIMD == 1
    #include <immintrin.h>
    #include <omp.h>
    #define VAR float
    #define FLOAT

#elif USE_SIMD == 2
    #include <mkl.h>
    #define VAR double
    #define DOUBLE
#else 
    #define NB_THREAD 0
    #define VAR float
    #define FLOAT

#endif
// #include <cblas.h>
#include <hdf5.h>


struct Matrix
{
    VAR* array;
    int *shape;
    int size_shape;
    int size_array;
    void (*Free)(struct Matrix* this);
    void (*Print)(struct Matrix* this);
    void (*Print_Shape)(struct Matrix* this);
    struct Matrix* (*__Copy)(struct Matrix this);
    void (*Copy)(struct Matrix* this, struct Matrix* other);
    struct Matrix* (*__Transpose)(struct Matrix this, int p1, int p2);
    void (*Transpose)(struct Matrix* this, int p1, int p2);
    struct Matrix* (*__Concat)(struct Matrix m1, struct Matrix m2, int axis);
    void (*Concat)(struct Matrix* m1, struct Matrix* m2, int axis);
    struct Matrix* (*__Add_Scalar)(struct Matrix, VAR scalar);
    void (*Add_Scalar)(struct Matrix*, VAR scalar);
    struct Matrix* (*__Mult_Scalar)(struct Matrix, VAR scalar);
    void (*Mult_Scalar)(struct Matrix*, VAR scalar);    

    void (*Save)(struct Matrix* this, hid_t file_id, char* name);
    
    void (*Mult)(struct Matrix* m1, struct Matrix* m2); 
    struct Matrix* (*__Mult)(struct Matrix m1, struct Matrix m2);
    void (*Div)(struct Matrix* m1, struct Matrix* m2);
    struct Matrix* (*__Div)(struct Matrix m1, struct Matrix m2);
    void (*Add )(struct Matrix* m1, struct Matrix* m2);
    struct Matrix* (*__Add)(struct Matrix m1, struct Matrix m2);
    void (*Sub )(struct Matrix* m1, struct Matrix* m2);
    struct Matrix* (*__Sub)(struct Matrix m1, struct Matrix m2);
    struct Matrix* (*__Dot)(struct Matrix m1, struct Matrix m2);
    void (*Dot)(struct Matrix* m1, struct Matrix* m2);

    struct Matrix* (*__Reshape)(struct Matrix this, int* new_shape, int size_new_shape);
    void (*Reshape)(struct Matrix* this, int* new_shape, int size_new_shape);

    struct Matrix* (*__Slicing)(struct Matrix this,int axis, int start, int end);
    void (*Slicing)(struct Matrix* this, int axis, int start, int end);

    void (*Put)(struct Matrix* this, struct Matrix* m2, int axis, int start, int end);

    struct Matrix* (*__Expand)(struct Matrix this, int shape, int size_shape);
    void (*Expand)(struct Matrix* this, int shape, int size_shape);

    struct Matrix* (*__Expand_Dim)(struct Matrix this, int axis);
    void (*Expand_Dim)(struct Matrix* this, int axis);

    struct Matrix* (*__Exponential)(struct Matrix this);
    void (*Exponential)(struct Matrix* this);

    struct Matrix* (*__Logarithm)(struct Matrix this);
    void (*Logarithm )(struct Matrix* this);

    struct Matrix* (*__Power)(struct Matrix this, VAR power);
    void (*Power)(struct Matrix* this, VAR power);

    struct Matrix* (*__Sum)(struct Matrix this, int axis);
    void (*Sum)(struct Matrix* this, int axis);

    struct Matrix* (*__Abs)(struct Matrix this);
    void (*Abs)(struct Matrix* this);
    
    VAR (*Max)(struct Matrix* this);
    VAR (*Min)(struct Matrix* this);
    
    struct Matrix* (*__Argmax)(struct Matrix this, int axis);
    void (*Argmax)(struct Matrix* this, int axis);

    struct Matrix* (*__Argmin)(struct Matrix this, int axis);
    void (*Argmin)(struct Matrix* this, int axis);


};
typedef struct Matrix Matrix;
Matrix* Init_Matrix(int* shape, int size, char* value);
Matrix* Load_Matrix(hid_t file_id, char* name);

#include "math/Multiplication.h"
#include "math/Transpose.h"
#include "math/Division.h"
#include "math/Substraction.h"
#include "math/Addition.h"
#include "math/Math_Utils.h"
#include "math/Logarithm.h"
#include "math/Exponential.h"
#include "math/Power.h"
#include "math/Sum.h"
#include "math/Abs.h"
#include "utils/Expand.h"
#include "math/min_max/Max.h"
#include "math/min_max/Min.h"
#include "utils/Copy.h"
#include "utils/Reshape.h"
#include "utils/Slicing.h"
#include "utils/Concatenate.h"

#endif
