CC      ?= cc
CFLAGS  ?= -std=c11 -O3 -march=native -Wall -Wextra
LDFLAGS ?= -lm

ifeq ($(OMP),1)
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
    CFLAGS  += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
    LDFLAGS += -L$(OMP_PREFIX)/lib -lomp
  else
    CFLAGS  += -fopenmp
    LDFLAGS += -fopenmp
  endif
else
  CFLAGS += -Wno-unknown-pragmas
endif

TARGET = siam

.PHONY: all clean

all: $(TARGET)

$(TARGET): siam.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
