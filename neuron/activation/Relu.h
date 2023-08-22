#ifndef RELU_H
#define RELU_H
#include <stddef.h>
#include "../../matrix/Matrix.h"
#include "../Layer.h"

struct Relu{
    Matrix* (*Forward)(struct Relu* relu, Matrix* input);
    Matrix* (*Backprop)(struct Relu* relu, Matrix* output, VAR learning_rate);
    void (*Initialize)(struct Relu* relu, Matrix* input_matrix);  
    void (*Free)(struct Relu* relu);
    Matrix* input ;
    int input_size;
    int output_size;
};
typedef struct  Relu Relu;
struct Layer;
struct Layer* Init_Relu();
#endif