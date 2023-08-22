#ifndef LOSS_H
#define LOSS_H
#include <stddef.h>
#include "../matrix/Matrix.h"
#include "loss_function/Mse.h"
#include "loss_function/Bce.h"
#include "loss_function/Cross_entropy.h"
#include <hdf5.h>
struct Loss{
    char* name;
    union{
        struct Mse* mse;
        struct Bce* bce;
        struct Cross_entropy* cross_entropy;
    }inheritance;
    VAR (*Loss)(struct Loss* loss, Matrix* y_true, Matrix* y_pred);
    Matrix* (*Loss_Prime)(struct Loss* loss, Matrix* y_true, Matrix* y_pred);
    void (*Free)(struct Loss* loss);
};
typedef struct Loss Loss;
Loss* Init_Loss();
Loss* Get_Loss(char* name);

#endif