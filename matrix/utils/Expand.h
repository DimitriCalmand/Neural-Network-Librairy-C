#ifndef EXPAND_H
#define EXPAND_H

Matrix* __Expand_Matrix(Matrix this, int shape, int size_shape);
void Expand_Matrix(Matrix* this, int shape, int size_shape);
Matrix* __Extand_First_Axis(Matrix m_1, Matrix m_2);
void Expand_Dim_Matrix(Matrix* m1, int axis);
Matrix* __Expand_Dim_Matrix(Matrix m, int axis);

#endif