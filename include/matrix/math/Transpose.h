#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "../Matrix.h"
void Transpose_Matrix(Matrix* this, int p1, int p2);
Matrix* __Transpose_Matrix(Matrix this, int p1, int p2);

void __Transpose_Matrix_Place(Matrix this, int p1, int p2, Matrix* res);

#endif