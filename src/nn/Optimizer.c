#include "nn/Optimizer.h"

void Free_Optimizer(Optimizer* this)
{
    if (strcmp(this->name, "Adam") == 0 && this->principal == 0)
    {
        this->m->Free(this->m);
        this->v->Free(this->v);
    }
    if (this->principal == 1)
    {
            free(this->parameters);
            free(this->values);
    }

    free(this);
}
void Adam(Optimizer* this, Matrix* gradient)
{
    this->iteration++;
    // this->weights->Print(this->weights);
    for (int i = 0; i!= this->weights->size_array; i++)
    {
        this->m->array[i] = this->values[0]*this->m->array[i] + (1-this->values[0])*gradient->array[i];
        this->v->array[i] = this->values[1]*this->v->array[i] + (1-this->values[1])*gradient->array[i]*gradient->array[i];
        VAR m_hat = this->m->array[i]/(1-pow(this->values[0], this->iteration));
        VAR v_hat = this->v->array[i]/(1-pow(this->values[1], this->iteration));
        this->weights->array[i] -= this->learning_rate*m_hat/(sqrt(v_hat)+this->values[2]);
    }
}
void SGD(Optimizer* this, Matrix* gradient)
{
    for (int i = 0; i!= this->weights->size_array; i++)
    {
        this->weights->array[i] -= this->learning_rate*gradient->array[i];
    }
}
void Optimize(Optimizer* this, Matrix* gradients)
{
    if (strcmp(this->name, "Adam") == 0)
    {
        Adam(this, gradients);
    }
    else if (strcmp(this->name, "SGD") == 0)
    {
        SGD(this, gradients);
    }
}
Optimizer* Create_Optimizer(Optimizer* this, Matrix* weights)
{
    Optimizer* optimizer = malloc(sizeof(Optimizer));
    optimizer->name = this->name;
    optimizer->nbr_parameters = this->nbr_parameters;
    optimizer->parameters = this->parameters;
    optimizer->values = this->values;
    optimizer->learning_rate = this->learning_rate;
    optimizer->principal = 0;
    optimizer->weights = weights;
    optimizer->iteration = this->iteration;
    if (strcmp(this->name, "Adam") == 0)
    {
        optimizer->m = Init_Matrix(weights->shape, weights->size_shape, "0");
        optimizer->v = Init_Matrix(weights->shape, weights->size_shape, "0");
    }
    // optimizer->Create_Optimizer = Create_Optimizer;
    optimizer->Free = Free_Optimizer;
    optimizer->Optimize = Optimize;
    return optimizer;

}
void Add_Function_Optimizer(Optimizer* optimizer)
{
    optimizer->Create_Optimizer = Create_Optimizer;
    optimizer->Free = Free_Optimizer;
    optimizer->Optimize = Optimize;
}
Optimizer* Init_Optimizer(char* name, int nbr_parameters, char** parameters, 
                            VAR* values, VAR learning_rate
                            )
{
    Optimizer* optimizer = malloc(sizeof(Optimizer));
    optimizer->name = name;
    optimizer->nbr_parameters = nbr_parameters;
    optimizer->parameters = parameters;
    optimizer->values = values;
    optimizer->learning_rate = learning_rate;
    optimizer->principal = 1;
    optimizer->iteration = 0;
    Add_Function_Optimizer(optimizer);
    return optimizer;
}

Optimizer* Init_Adam(VAR learning_rate, VAR beta1, VAR beta2, VAR epsilon)
{
    char** parameters = malloc(3*sizeof(char*));
    parameters[0] = "beta1";
    parameters[1] = "beta2";
    parameters[2] = "epsilon";
    VAR* values = malloc(3*sizeof(VAR));
    values[0] = beta1;
    values[1] = beta2;
    values[2] = epsilon;
    return Init_Optimizer("Adam", 3, parameters, values, learning_rate);
}
Optimizer* Init_SGD(VAR learning_rate)
{
    return Init_Optimizer("SGD", 0, NULL, NULL, learning_rate);
}

