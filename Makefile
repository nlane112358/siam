CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
LDFLAGS ?= -lm
GO       ?= go

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

TARGET = bin/siam
GO_TARGETS = bin/asf bin/sarimg

.PHONY: all clean

all: $(TARGET) $(GO_TARGETS)

SOURCES = main.cpp
HEADERS = config.hpp image.hpp network.hpp trainer.hpp detector.hpp spectral.hpp temporal.hpp dataset.hpp

$(TARGET): $(SOURCES) $(HEADERS) | bin
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bin/asf: tools/asf.go | bin
	$(GO) build -o $@ $<

bin/sarimg: tools/sarimg.go | bin
	$(GO) build -o $@ $<

bin:
	mkdir -p $@

clean:
	rm -f $(TARGET) $(GO_TARGETS)
