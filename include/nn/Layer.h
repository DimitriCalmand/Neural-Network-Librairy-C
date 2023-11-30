#ifndef LAYER_H
#define LAYER_H
#include <stddef.h>
#include "matrix/Matrix.h"
#include "Dense.h"
#include "activation/Sigmoid.h"
#include "activation/Tanh.h"
#include "activation/Relu.h"
#include "activation/Softmax.h"
#include "activation/Leaky_relu.h"
#include "Embedding.h"
#include "Lstm.h"
#include <hdf5.h>
#include <string.h>
#include "Optimizer.h"

struct Layer 
{
    char *layer;
    union 
    {
        struct Dense* dense;
        struct Sigmoid* sigmoid;
        struct Tanh* tanh;
        struct Relu* relu;
        struct Softmax* softmax;
        struct Leaky_Relu* leaky_relu;
        struct Lstm* lstm;
        struct Embedding* embedding;
    }inheritance;
    Matrix* (*Forward)(struct Layer* this, Matrix* input);
    Matrix* (*Backprop)(struct Layer* this, Matrix* input, VAR learning_rate);
    void (*Initialize)(struct Layer* this, Matrix* input_matrix, Optimizer* Optimizer);
    void (*Free)(struct Layer* this);
    void (*Save)(struct Layer* this, hid_t file_id);
    Matrix* (*Get_Input)(struct Layer* this);
};
typedef struct Layer Layer;
Layer* Init_Layer();
Layer* Load_Layer(hid_t file_id, Optimizer* optimizer);
#endif
