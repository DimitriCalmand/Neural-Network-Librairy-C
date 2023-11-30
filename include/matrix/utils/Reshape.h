#ifndef RESHAPE_H
#define RESHAPE_H

#include "../Matrix.h"

void Reshape_Matrix(Matrix* this, int* new_shape, int size_new_shape);
Matrix* __Reshape_Matrix(Matrix this, int* new_shape, int size_new_shape);

#endif  