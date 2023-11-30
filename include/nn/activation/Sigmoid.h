#ifndef SIGMOID_H
#define SIGMOID_H
#include <stddef.h>
#include "matrix/Matrix.h"
#include "../Layer.h"

struct Sigmoid{
    Matrix* (*Forward)(struct Sigmoid* sigmoid, Matrix* input);
    Matrix* (*Backprop)(struct Sigmoid* sigmoid, Matrix* output, VAR learning_rate);
    void (*Initialize)(struct Sigmoid* sigmoid, Matrix* input_matrix);  
    void (*Free)(struct Sigmoid* sigmoid);
    void (*Save)(struct Sigmoid* this, hid_t file_id);
    
    Matrix* input ;
    int input_size;
    int output_size;
};
typedef struct  Sigmoid Sigmoid;
struct Layer;
struct Layer* Init_Sigmoid();
struct Layer* Load_Sigmoid(hid_t file_id __attribute__((unused)));
#endif
