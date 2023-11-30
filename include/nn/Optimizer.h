#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "matrix/Matrix.h"
#include <string.h>
#include <stdlib.h>

#include <math.h>
struct Optimizer
{
    char* name;
    char** parameters;
    VAR* values;
    int nbr_parameters;
    VAR learning_rate;
    Matrix* m;
    Matrix* v;
    Matrix* weights;
    int principal;
    int iteration;
    struct Optimizer* (*Create_Optimizer)(struct Optimizer* this, Matrix* weights);
    void (*Free)(struct Optimizer* this);
    void (*Optimize)(struct Optimizer* this, Matrix* gradients);
};
typedef struct Optimizer Optimizer;
Optimizer* Init_Optimizer(char* name, int nbr_parameters, char** parameters, 
                            VAR* values, VAR learning_rate
                            );
Optimizer* Init_Adam(VAR learning_rate, VAR beta1, VAR beta2, VAR epsilon);
Optimizer* Init_SGD(VAR learning_rate);


#endif
