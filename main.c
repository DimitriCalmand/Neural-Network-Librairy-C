    #include <stdio.h>
#include <stdlib.h>
#include "matrix/Matrix.h"
#include "neuron/Loss.h"
#include "neuron/Model.h"  
#include "prepare_dataset.h" 
#include "neuron/Saver.h"
#include <time.h>
#include <string.h>
#include <sys/time.h>

void generate_sequences(int batch_size, int nbr_class, int size, Matrix* x, Matrix* y)
{
    for (int i = 0; i != batch_size; i++)
    {
        int pos_i = i*size*nbr_class;
        int start =  rand() % (nbr_class-size);
        x->array[pos_i+ start] = 1;
        // y->array[i*nbr_class+ start+size] = 1;
        int step = rand()%2+1;
        y->array[pos_i+ start+step] = 1;
        for (int j = 1; j != size; j++)
        {
            x->array[pos_i + j*nbr_class + start+j*step] = 1;
            y->array[pos_i + j*nbr_class + start+(j+1)*step] = 1;
        }
    }
}
int main()
{   
    // mkl_set_num_threads(1);
    Matrix* x = Init_Matrix((int[]){100, 10, 20}, 3, "0");
    Matrix* y = Init_Matrix((int[]){100, 10, 20}, 3, "0");

    generate_sequences(100, 20, 10, x, y);
    Model *model = Init_Model((int[]){10, 20}, 2);
    model->Add(model, Init_Lstm(50, 1));
    model->Add(model, Init_Dense(20));
    model->Add(model, Init_Sigmoid()); 
    model->Compile(model, Init_Bce(), Init_SGD(0.5), 1.);  
    model->Summary(model);       

    // Model* model = Load_Model("./model.h5");
    model->Fit(model, x, y, x, y, 400, 100, 50);

    // Save(model, "./model.h5"); 

    x->Free(x);
    y->Free(y);
    model->Free(model);

}

