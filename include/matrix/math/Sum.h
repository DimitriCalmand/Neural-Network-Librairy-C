#ifndef SUM_H
#define SUM_H

#include "../Matrix.h"
void Sum_Matrix(Matrix* m1, int axis);
Matrix* __Sum_Matrix(Matrix m1, int axis);
void __Sum_Matrix_Place(Matrix this, int axis, Matrix* res);
#endif