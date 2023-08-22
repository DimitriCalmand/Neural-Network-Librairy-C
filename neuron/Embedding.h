#ifndef EMBEDDING_H
#define EMBEDDING_H
#include <stddef.h>
#include "../matrix/Matrix.h"
#include "Layer.h"
#include "Optimizer.h"

struct Embedding {
    Matrix* (*Forward)(struct Embedding* embedding, Matrix* input);
    Matrix* (*Backprop)(struct Embedding* embedding, Matrix* output);
    void (*Initialize)(struct Embedding* embedding, Matrix* input_matrix, Optimizer* optimizer);  
    void (*Free)(struct Embedding* embedding); 
    void (*Save)(struct Embedding* embedding, hid_t file_id);
    Matrix* weights ;
    Matrix* input;
    int vocab_size;
    int embedding_dim;
    Optimizer* Optimizer_weights;
};
struct Layer;
typedef struct Embedding Embedding ;

struct Layer* Init_Embedding(int voacb_size, int embedding_dim);
struct Layer* Load_Embedding(hid_t file, Optimizer* optimizer);

#endif