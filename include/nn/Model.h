#ifndef MODEL_H
#define MODEL_H
#include <stddef.h>
#include "matrix/Matrix.h"   
#include "Loss.h"
#include "Layer.h"
#include "Model.h"
#include "Dense.h"
#include "activation/Sigmoid.h"
#include "activation/Relu.h"
#include "activation/Softmax.h"
#include "activation/Tanh.h"
#include "loss/Mse.h"
#include "loss/Bce.h"
#include <sys/time.h>

struct Model
{
    void (*Compile)(struct Model* model, Loss* loss, Optimizer* optimizer, VAR learning_rate);
    void (*Fit)(struct Model* model, Matrix* x_train, Matrix* y_train,
                 Matrix* x_test, Matrix* y_test,  int epoch,
                 int batch_size, int verbose );
    void (*Add)(struct Model* model, Layer* layer);
    Matrix* (*Predict)(struct Model* model, Matrix* x_train);
    VAR (*Accuracy)(struct Model* model, Matrix* x_train, Matrix* y_train);
    void (*Free)(struct Model* model);
    void(*Summary)(struct Model* model);
    int* shape_input;
    int size_shape_input;
    Layer** layers;
    Loss* loss;
    int is_compiled;
    int nb_layers;
    double learning_rate ;
    Optimizer* optimizer;
};
typedef struct Model Model;
Model* Init_Model(int* shape_input, int size_shape_input);
#endif    
