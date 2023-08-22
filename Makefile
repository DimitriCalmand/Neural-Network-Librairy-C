CC = gcc


CFLAGS = -Wall -Wextra \
         -std=c99 \
         -O2 \
         -g \
         -mavx512dq \
         -DUSE_SIMD=1 \
         -D_GNU_SOURCE

LDFLAGS = -g

LDLIBS = -lm \
         -lhdf5 \
         -fuse-ld=gold \
         -fsanitize=address

TARGET ?= main

SRCS=$(wildcard ./main.c)
SRCS+= $(wildcard ./prepare_dataset.c)
SRCS+= $(wildcard ./matrix/*.c)
SRCS+= $(wildcard ./matrix/math/*.c)
SRCS+= $(wildcard ./matrix/utils/*.c)
SRCS+= $(wildcard ./matrix/math/min_max/*.c)
SRCS+= $(wildcard ./neuron/*.c)
SRCS+= $(wildcard ./neuron/activation/*.c)
SRCS+= $(wildcard ./neuron/loss_function/*.c)

OBJS=$(SRCS:.c=.o)

all: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) ${LDFLAGS} ${LDLIBS} -o $(TARGET)

clean:
	${RM} ${OBJS}  ${DEPS} ${TARGET}
list:
	@echo $(SRCS)
