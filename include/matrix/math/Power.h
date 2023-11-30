#ifndef POWER_H
#define POWER_H

#include "../Matrix.h"
void Power_Matrix(Matrix* m1, VAR power);
Matrix* __Power_Matrix(Matrix m1, VAR power);
void __Power(Matrix* this, VAR power, Matrix* res);

#endif