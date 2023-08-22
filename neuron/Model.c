#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "Model.h"

void Add(Model* this, Layer* layer)
{
    this->layers = realloc(this->layers,(++this->nb_layers)*sizeof(Layer*));
    this->layers[this->nb_layers-1] = layer;
}
void Compile(Model* this, Loss* loss, Optimizer* optimzer, VAR learning_rate)
{
    this->loss = loss;
    this->is_compiled = 1;
    this->learning_rate = learning_rate;
    this->optimizer = optimzer;
    // Create a shape for the input with a batch size of 1
    int* shape = malloc((this->size_shape_input+1)*sizeof(int));
    shape[0] = 1;
    for (int i = 1; i!=this->size_shape_input+1;i++){
        shape[i] = this->shape_input[i-1];
    }
    Matrix* train = Init_Matrix(shape, this->size_shape_input+1, "1");
    for (int i = 0; i!=this->nb_layers;i++)
    {
        this->layers[i]->Initialize(this->layers[i], train, optimzer);
        Matrix* tmp_forward = this->layers[i]->Forward(this->layers[i], train);
        train->Copy(train, tmp_forward);
        tmp_forward->Free(tmp_forward);
    }
    
    free(shape);
    train->Free(train);
}
Matrix* Predict(Model* this, Matrix* x_train){
    Matrix* input = x_train->__Copy(*x_train);
    for (int k = 0; k!=this->nb_layers; k++)
    {
        // printf("%s\n", this->layers[k]->layer);
        Matrix* tmp = this->layers[k]->Forward(this->layers[k], input);
        input->Copy(input, tmp);
        // printf("exit\n");
        // input->Print(input);
        tmp->Free(tmp);
    }
    return input;
}

VAR __Accuracy(Model* this __attribute__((unused)), Matrix* prediction, Matrix* y_train)
{
    int nb_correct = 0;
    if (prediction->shape[prediction->size_shape-1] > 1)
    {
        Matrix* prediction_max = prediction->__Argmax(*prediction, -1);
        Matrix* true_max = y_train->__Argmax(*y_train, -1);
        int size = true_max->size_array;
        for (int i = 0; i!= size; i++)
        {
            if (true_max->array[i] == prediction_max->array[i])
            {
                nb_correct++;
            }
        } 
        true_max->Free(true_max);
        prediction_max->Free(prediction_max);
        // prediction->Free(prediction);
        return (VAR)nb_correct/size;
    }
    for (int i = 0; i!=prediction->size_array; i++)
    {
        if (prediction->array[i] > 0.5 && y_train->array[i] == 1)
        {
            nb_correct++;
        }
        else if (prediction->array[i] < 0.5 && y_train->array[i] == 0)
        {
            nb_correct++;
        }
    }    
    return (VAR)nb_correct/y_train->size_array;
}   
VAR Accuracy(Model* this , Matrix* x_train, Matrix* y_train)
{
    Matrix* prediction = Predict(this, x_train);
    VAR res =  __Accuracy(this, prediction, y_train);
    prediction->Free(prediction);
    return res;
}

void __select_color_by_value(VAR value, VAR max_value){
    VAR ratio = value/max_value;
    int red = 255*(1-ratio);
    int green = 255*ratio;
    printf("\033[0;38;2;%d;%d;0m", red, green);
}
void __Test(Model* this, Matrix* prediction_train, Matrix* y_train,
            Matrix* x_test, Matrix* y_test, VAR end_time,
            int i, int epoch)
{
    Matrix* prediction_test = Predict(this, x_test);
    VAR loss_test = this->loss->Loss(this->loss, y_test, prediction_test);
    VAR accuracy_test = __Accuracy(this, prediction_test, y_test);
    VAR loss_train = this->loss->Loss(this->loss, y_train, prediction_train);
    VAR accuracy_train = __Accuracy(this, prediction_train, y_train);
    printf("Epoch %d/%d, Time %f", i, epoch, end_time);
    printf(", Loss train : ");   
    printf("\033[0;33m"); 
    printf("%f", loss_train);
    printf("\033[0m");
    printf(", Accuracy train : ");
    __select_color_by_value(accuracy_train, 1);
    printf("%f", accuracy_train);
    printf("\033[0m");
    printf(", Loss test : ");
    printf("\033[0;33m");
    printf("%f", loss_test);
    printf("\033[0m");
    printf(", Accuracy test : ");
    __select_color_by_value(accuracy_test, 1);
    printf("%f\n", accuracy_test);
    printf("\033[0m");
    prediction_test->Free(prediction_test);
    // prediction_train->Free(prediction_train);
}
Matrix* __Training(Model* this, Matrix* X, Matrix* y)
{
    /*      Forward Propagation     */

    Matrix* prediction = Predict(this, X);
    Matrix* loss_prime = this->loss->Loss_Prime(this->loss, y, prediction);
    // prediction->Print(prediction);
    /*      Backpropagation     */
    for (int k = this->nb_layers-1; k!=-1; k--)
    {
        
        Matrix* tmp = this->layers[k]->Backprop(this->layers[k], loss_prime, this->learning_rate);
        loss_prime->Copy(loss_prime, tmp);
        tmp->Free(tmp);
    }
    loss_prime->Free(loss_prime);
    // prediction->Free(prediction);
    return prediction;
}
void Fit (Model* this, Matrix* x_train, Matrix* y_train, 
        Matrix* x_test, Matrix* y_test,
        int epoch, int batch_size, int verbose)
{
    
    clock_t start = clock();
    Matrix* pred = Init_Matrix((int[]){1}, 1, "null");
    Matrix* y_true = Init_Matrix((int[]){1}, 1, "null");
    for (int i = 0; i != epoch; i++)
    {
        int j = 0;
        for (; j<=x_train->shape[0]-batch_size; j+=batch_size)
        {
            // struct timeval start_time, end_time;
            // double elapsed_time; 

            // gettimeofday(&start_time, NULL); // record start time
            Matrix* X = x_train->__Slicing(*x_train, 0, j, j+batch_size);
            Matrix* y = y_train->__Slicing(*y_train, 0, j, j+batch_size);
            Matrix* tmp = __Training(this, X, y);
            pred->Copy(pred, tmp);
            y_true->Copy(y_true, y);
            tmp->Free(tmp);
            X->Free(X);
            y->Free(y);
            // gettimeofday(&end_time, NULL); // record end time
            // elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
            //                 (end_time.tv_usec - start_time.tv_usec) / 1000000.0;

            // VAR end_time = (VAR)(clock() - start)/CLOCKS_PER_SEC;
            // __Test(this, pred, y_true, x_test, y_test, end_time, i, epoch);
            // start = clock();
        }
        if (j!=x_train->shape[0])
        {
            Matrix* X = x_train->__Slicing(*x_train, 0, j, x_train->shape[0]);
            Matrix* y = y_train->__Slicing(*y_train, 0, j, x_train->shape[0]);
            Matrix* useless_pred = __Training(this, X, y);
            useless_pred->Free(useless_pred);
            X->Free(X);
            y->Free(y);
        }
        if (i % verbose == 0)
        {
            VAR end_time = (VAR)(clock() - start)/CLOCKS_PER_SEC;
            __Test(this, pred, y_true, x_test, y_test, end_time, i, epoch);
            start = clock();
        }
    }
    pred->Free(pred);
    y_true->Free(y_true);
}
char* shape_to_string(int* shape, int size)
{
    // calculate the size of the required string
    // I use 10 because a digit is on 32 bit so 10 digit max
    int str_size = size * 10 + size + 1;  
    char* str = malloc(str_size * sizeof(char));
    int pos = 0;
    
    pos += sprintf(&str[pos], "(");
    
    for (int i = 0; i < size; i++) {
        pos += sprintf(&str[pos], "%d", shape[i]);
        if (i != size - 1) {
            pos += sprintf(&str[pos], ",");
        }
    }
    
    pos += sprintf(&str[pos], ")");
    str[pos] = '\0';
    
    return str;
}

void Summary(Model* this)
{
    int first_col = 40;
    int size_row = 80;
    char layer_name[] = "layer00";
    char* line = malloc(size_row*sizeof(char));
    for (int i = 0; i < size_row; i++) 
    {
        line[i] = '=';
    }
    char* type = "layer : type";
    char* shape = "output shape";
    printf("\n%s%*c%s\n%s\n", type, (int)(first_col - strlen(type)), ' ', shape, line);
    for (int i = 0; i < size_row; i++) 
    {
        line[i] = '-';
    }

    for (int i = 0; i< this->nb_layers; i++)
    {
        layer_name[5] = i/10+'0';
        layer_name[6] = i%10+'0';
        char* type_layer = this->layers[i]->layer;
        Matrix* input = this->layers[i]->Get_Input(this->layers[i]);
        char* shape_test = shape_to_string(input->shape, input->size_shape);
        printf("%s : %s%*c%s\n%s\n", layer_name, type_layer, (int)(first_col - strlen(layer_name)- strlen(type_layer) - 3), ' ', shape_test, line);
        free(shape_test);
    }
    free(line);
    printf("\n");

}
void Free_Model(Model* this)
{
    for (int i = 0; i!=this->nb_layers; i++)
    {
        this->layers[i]->Free(this->layers[i]);
    }
    if (this->is_compiled == 1){
        this->loss->Free(this->loss);
    }
    free(this->layers);
    this->optimizer->Free(this->optimizer);
    free(this);

}
Model* Init_Model(int* shape_input, int size_shape_input)
{
    Model* model = malloc(sizeof(Model));
    model->Add = Add;
    model->Compile = Compile;
    model->Fit = Fit;
    model->Free = Free_Model;
    model->layers = NULL;
    model->Predict = Predict;
    model->Summary = Summary;
    model->nb_layers = 0;
    model->is_compiled = 0;
    model->shape_input = shape_input;
    model->size_shape_input = size_shape_input;
    model->learning_rate = 0.9;
    return model;
}