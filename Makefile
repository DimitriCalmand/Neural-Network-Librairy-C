CC = gcc

CFLAGS = -Wall -Wextra \
         -std=c99 \
		 -I./include \
		 -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
         -lhdf5 \
	     -g \
		 -DUSE_SIMD=0 \
         -fsanitize=address \
         -mavx512dq \
         -D_GNU_SOURCE

LDLIBS = -lm \
         -fuse-ld=gold \

LIBS = $(wildcard lib/*.a)

TARGET ?= main
LIBRARY = libnn.a

OBJS_MAIN = main.o src/dataset/prepare_dataset.o

SRCS+= $(wildcard ./src/nn/*.c)
SRCS+= $(wildcard ./src/nn/activation/*.c)
SRCS+= $(wildcard ./src/nn/loss/*.c)

OBJS=$(SRCS:.c=.o)

all: $(OBJS_MAIN) $(OBJS) $(LIBS)
	$(CC) $< -o $(TARGET) $(LIBS) $(OBJS) $(CFLAGS) $(LDLIBS)

library: $(OBJS)
	ar -crs $(LIBRARY) $(OBJS)
clean:
	${RM} ${OBJS}  ${DEPS} ${TARGET} $(LIBRARY) $(OBJS_MAIN)

list:
	@echo $(SRCS) $(LIBS)
