#ifndef LEAKY_RELU_H
#define LEAKY_RELU_H
#include <stddef.h>
#include "matrix/Matrix.h"
#include "../Layer.h"

struct Leaky_Relu{
    Matrix* (*Forward)(struct Leaky_Relu* leaky_relu, Matrix* input);
    Matrix* (*Backprop)(struct Leaky_Relu* leaky_relu, Matrix* output, VAR learning_rate);
    void (*Initialize)(struct Leaky_Relu* leaky_relu, Matrix* input_matrix);  
    void (*Free)(struct Leaky_Relu* leaky_relu);
    Matrix* input ;
    int input_size;
    int output_size;
};
typedef struct  Leaky_Relu Leaky_Relu;
struct Layer;
struct Layer* Init_Leaky_Relu();
#endif
