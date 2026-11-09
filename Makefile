CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS ?= -lm

ifeq ($(OMP),1)
  UNAME_S := $(shell uname -s)
  ifeq ($(UNAME_S),Darwin)
    OMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
    CXXFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
    LDFLAGS += -L$(OMP_PREFIX)/lib -lomp
  else
    CXXFLAGS += -fopenmp
    LDFLAGS += -fopenmp
  endif
else
  CXXFLAGS += -Wno-unknown-pragmas
endif

TARGET = siam

.PHONY: all clean

all: $(TARGET)

SOURCES = main.cpp
HEADERS = config.hpp image.hpp network.hpp trainer.hpp detector.hpp spectral.hpp dataset.hpp

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
