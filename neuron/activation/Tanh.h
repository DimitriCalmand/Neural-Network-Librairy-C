#ifndef TANH_H
#define TANH_H
#include <stddef.h>
#include "../../matrix/Matrix.h"
#include "../Layer.h"

struct Tanh{
    Matrix* (*Forward)(struct Tanh* tanh, Matrix* input);
    Matrix* (*Backprop)(struct Tanh* tanh, Matrix* output, VAR learning_rate);
    void (*Initialize)(struct Tanh* tanh, Matrix* input_matrix);  
    void (*Free)(struct Tanh* tanh);
    Matrix* input ;
    int input_size;
    int output_size;
};
typedef struct  Tanh Tanh;
struct Layer;
struct Layer* Init_Tanh();
#endif