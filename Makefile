CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -Wall -Wextra -Wno-unknown-pragmas -Iinclude
LDFLAGS ?= -lm

SRCS    := src/image.c src/network.c src/dataset.c src/trainer.c src/detector.c src/main.c
OBJS    := $(SRCS:.c=.o)
TARGET  := pgmcd

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
