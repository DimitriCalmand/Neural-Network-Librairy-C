#ifndef MSE_H
#define MSE_H 
#include <stddef.h>
#include "../../matrix/Matrix.h"
#include "../Loss.h"

struct Mse{
    VAR (*Loss)(Matrix* y_true, Matrix* y_pred);
    Matrix* (*Loss_Prime)(Matrix* y_true, Matrix* y_pred);
    void (*Free)(struct Mse* mse);
};
typedef struct Mse Mse;
struct Loss* Init_Mse();
#endif