#ifndef DENSE_H 
#define DENSE_H 
#include <stddef.h>
#include "../matrix/Matrix.h"
#include "Layer.h"
#include "Optimizer.h"
struct Dense {
    Matrix* (*Forward)(struct Dense* dense, Matrix* input);
    Matrix* (*Backprop)(struct Dense* dense, Matrix* output);
    void (*Initialize)(struct Dense* dense, Matrix* input_matrix, Optimizer* Optimizer);  
    void (*Save)(struct Dense* dense, hid_t file_id);
    void (*Free)(struct Dense* dense); 
    void (*Load)(struct Dense* dense, hid_t file_id);
    Matrix* weights ;
    Matrix* bias ;
    Matrix* input;
    int output_size;
    Optimizer* Optimizer_weights;
    Optimizer* Optimizer_bias;
};
struct Layer;
typedef struct Dense Dense ;
struct Layer* Init_Dense(int size_input);
struct Layer* Load_Dense(hid_t file_id, Optimizer* optimizer);
#endif