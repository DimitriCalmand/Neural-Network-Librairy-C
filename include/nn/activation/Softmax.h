#ifndef SOFTMAX_H
#define SOFTMAX_H
#include <stddef.h>
#include "matrix/Matrix.h"
#include "../Layer.h"

struct Softmax{
    Matrix* (*Forward)(struct Softmax* softmax, Matrix* input);
    Matrix* (*Backprop)(struct Softmax* softmax, Matrix* output, VAR learning_rate);
    void (*Initialize)(struct Softmax* softmax, Matrix* input_matrix);  
    void (*Free)(struct Softmax* softmax);
    Matrix* input ;
    int input_size;
    int output_size;
};
typedef struct  Softmax Softmax;
struct Layer;
struct Layer* Init_Softmax();
#endif
