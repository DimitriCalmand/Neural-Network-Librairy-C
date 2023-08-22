#ifndef ADDITION_H
#define ADDITION_H
#include "../Matrix.h"

void Add_Scalar_Matrix(Matrix* this, VAR scalar);
Matrix* __Add_Scalar_Matrix(Matrix this, VAR scalar);

void Add_Matrix(Matrix* m1, Matrix* m2);
Matrix* __Add_Matrix(Matrix m1, Matrix m2);

void __Add_Mat_Place(Matrix m_1, Matrix m_2, Matrix* res);
#endif