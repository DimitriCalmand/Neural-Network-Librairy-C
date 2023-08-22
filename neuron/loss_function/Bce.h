#ifndef BCE_H
#define BCE_H 
#include <stddef.h>
#include "../../matrix/Matrix.h"
#include "../Loss.h"

struct Bce{
    VAR (*Loss)(Matrix* y_true, Matrix* y_pred);
    Matrix* (*Loss_Prime)(Matrix* y_true, Matrix* y_pred);
    void (*Free)(struct Bce* bce);
};
typedef struct Bce Bce;
struct Loss* Init_Bce();
#endif