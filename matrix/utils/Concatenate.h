#ifndef CONCATENATE_H
#define CONCATENATE_H
#include "../Matrix.h"
Matrix* __Concat_Matrix(Matrix m1, Matrix m2, int axis);
void Concat_Matrix(Matrix* m1, Matrix* m2, int axis);
#endif  