#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H 
#include <stddef.h>
#include "../../matrix/Matrix.h"
#include "../Loss.h"

struct Cross_entropy{
    VAR (*Loss)(Matrix* y_true, Matrix* y_pred);
    Matrix* (*Loss_Prime)(Matrix* y_true, Matrix* y_pred);
    void (*Free)(struct Cross_entropy* mse);
};
typedef struct Cross_entropy Cross_entropy;
struct Loss* Init_Cross_entropy();
#endif