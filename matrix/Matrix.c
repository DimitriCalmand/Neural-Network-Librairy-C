#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Matrix.h"
#include <string.h>
#include <time.h>


void Print_Matrix(Matrix* this)
{
    int size_shape = this->size_shape;
    if (size_shape==1)
    {
        for(int i = 0; i!=this->shape[0]; i++)
        {
            printf("  %f",this->array[i]);
        }
    }
    else if (size_shape==2)
    {
        int rows = this->shape[size_shape-2];
        int cols = this->shape[size_shape-1];
        for (int i = 0; i!=rows; i++)
        {
            printf("\n");
            for (int j = 0; j!=cols; j++)
            {
                printf("  %f",this->array[i*cols+j]);
            }
        }
    }
    else {
        int dims = this->shape[size_shape-3];
        int rows = this->shape[size_shape-2];
        int cols = this->shape[size_shape-1];
        for (int k = 0; k!=dims; k++)
        {
            printf("Matrice number %d",k);
            for (int i = 0; i!=rows; i++)
            {
                printf("\n");
                for (int j = 0; j!=cols; j++)
                {
                    printf("  %f",this->array[k*rows*cols+i*cols+j]);
                }
            }
            printf("\n");
        }
    }
    printf("\n");
}
void Print_Shape(Matrix* this)
{
    printf("Shape: ");
    for (int i = 0; i!=this->size_shape; i++)
    {
        printf("%d ",this->shape[i]);
    }
    printf("\n");
}


void Free_Matrix(Matrix* this)
{
    free(this->array);
    free(this->shape);
    free(this);
}
void Save_Matrix(Matrix* this, hid_t file_id, char* name)
{
    hid_t matrix_group = H5Gcreate(file_id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Shape
    hsize_t dims[1] = {this->size_shape}; 
    hid_t shape_space = H5Screate_simple(1, dims, NULL);
    hid_t shape_dataset = H5Dcreate(matrix_group, "shape", H5T_NATIVE_INT, shape_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->shape);
    //Close
    H5Dclose(shape_dataset);
    H5Sclose(shape_space);

    // Array
    dims[0] = this->size_array;
    hid_t array_space = H5Screate_simple(1, dims, NULL);
    hid_t array_dataset = H5Dcreate(matrix_group, "array", H5T_NATIVE_DOUBLE, array_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(array_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, this->array);
    //Close
    H5Dclose(array_dataset);
    H5Sclose(array_space);

    //Close
    H5Gclose(matrix_group);
}
void Function_Matrix(Matrix* res)
{
    res->Print = Print_Matrix;
    res->Print_Shape = Print_Shape;
    res->Free = Free_Matrix;
    res->__Transpose = __Transpose_Matrix;
    res->Transpose = Transpose_Matrix;
    res->__Copy = __Copy_Matrix;
    res->Copy = Copy_Matrix;
    res->__Concat = __Concat_Matrix;
    res->Concat = Concat_Matrix;
    res->__Add_Scalar = __Add_Scalar_Matrix;
    res->Add_Scalar = Add_Scalar_Matrix; 
    res->__Mult_Scalar = __Mult_Scalar_Matrix;
    res->Mult_Scalar = Mult_Scalar_Matrix;  

    res->Save = Save_Matrix;
    res->__Mult = __Mult_Matrix;
    res->Mult = Mult_Matrix;
    res->__Div = __Div_Matrix;
    res->Div = Div_Matrix;
    res->__Add = __Add_Matrix;
    res->Add = Add_Matrix;
    res->__Sub = __Sub_Matrix;
    res->Sub = Sub_Matrix;
    
    res->Dot = Dot_Matrix;
    res->__Dot = __Dot_Matrix;   

    res->Reshape = Reshape_Matrix;
    res->__Reshape = __Reshape_Matrix;

    res->__Slicing = __Slicing_Matrix;
    res->Slicing = Slicing_Matrix;
    res->Put = Put_Matrix;

    res->Logarithm = Logarithm_Matrix;
    res->__Logarithm = __Logarithm_Matrix;

    res->Exponential = Exponential_Matrix;
    res->__Exponential = __Exponential_Matrix;

    res->Power = Power_Matrix;
    res->__Power = __Power_Matrix;

    res->Sum = Sum_Matrix;
    res->__Sum = __Sum_Matrix;

    res->Abs = Abs_Matrix;
    res->__Abs = __Abs_Matrix;

    res->__Expand = __Expand_Matrix;
    res->Expand = Expand_Matrix;    

    res->__Expand_Dim = __Expand_Dim_Matrix;
    res->Expand_Dim = Expand_Dim_Matrix;
    
    res->Max = Max_Matrix;
    res->Min = Min_Matrix;
    
    res->__Argmax = __Argmax_Matrix;
    res->Argmax = Argmax_Matrix;

    res->__Argmin = __Argmin_Matrix;
    res->Argmin = Argmin_Matrix;
}

Matrix* Load_Matrix(hid_t file_id, char* name)
{
    Matrix* loaded_matrix = malloc(sizeof(Matrix));

    hid_t matrix_group = H5Gopen(file_id, name, H5P_DEFAULT);

    // Shape
    hid_t shape_dataset = H5Dopen(matrix_group, "shape", H5P_DEFAULT);
    hid_t dataspace_size_shape = H5Dget_space(shape_dataset);
    hsize_t size_shape = H5Sget_simple_extent_npoints(dataspace_size_shape);
    loaded_matrix->size_shape = (int) size_shape;
    loaded_matrix->shape = malloc(sizeof(int) * loaded_matrix->size_shape);
    H5Dread(shape_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, loaded_matrix->shape);
    //Close
    H5Dclose(shape_dataset);

    // Array
    hid_t array_dataset = H5Dopen(matrix_group, "array", H5P_DEFAULT);
    hid_t dataspace_size_array = H5Dget_space(array_dataset);
    hsize_t size_array = H5Sget_simple_extent_npoints(dataspace_size_array);
    loaded_matrix->size_array = (int) size_array;
    loaded_matrix->array = malloc(sizeof(VAR) * loaded_matrix->size_array);
    H5Dread(array_dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, loaded_matrix->array);
    //Close
    H5Dclose(array_dataset);

    //Close
    H5Gclose(matrix_group);
    Function_Matrix(loaded_matrix);
    return loaded_matrix;
}

Matrix* Init_Matrix(int* shape, int size_shape, char* value)
{
    Matrix* res = malloc(sizeof(Matrix));
    res->shape = malloc(size_shape*sizeof(int));
    res->size_shape = size_shape;
    
    int size_array = 1;
    for(int i = 0; i!=size_shape; i++)
    {
        res->shape[i] = shape[i];
        size_array*=shape[i];
    }
    res->size_array = size_array; 
    // printf("size+array%d\n = ", size_array);
    res->array = calloc(size_array, sizeof(VAR));
    // res->array = malloc(size_array*sizeof(VAR));
    /* If value == 999, it is because I don't need a specific value */
    if (strcmp(value, "random") == 0)
    {
        srand(time(NULL));
        for (int i = 0; i!= size_array; i++)    
        {
            res->array[i] = ((VAR) (rand()%10000-5000))/10000;
        }
    }
    else if (strcmp(value, "range") == 0)
    {
        for (int i = 0; i!= size_array; i++)
        {
            res->array[i] =(VAR) i+1;
        }
    }
    else if (strcmp(value, "debug") == 0)
    {
        for (int i = 0; i!= size_array; i++)
        {
            res->array[i] = (VAR)i / 100.f;
        }
    }
    else if (strcmp(value, "null") != 0)
    {
        for (int i = 0; i!= size_array; i++)
        {
            res->array[i] = (VAR) atoi(value);
        }
    }

    // Function
    Function_Matrix(res);
    return res;
}
