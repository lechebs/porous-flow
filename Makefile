CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma

DEFINES =
SRC_DIR = src

.PHONY: clean

all: benchmark test

float: DEFINES += -DFLOAT
float: all

auto-vec: DEFINES += -DAUTO_VEC
auto-vec: all

SRC_FILES = $(addprefix $(SRC_DIR)/, equations.c finite-diff.c lin-solver.c)

benchmark: $(SRC_DIR)/benchmark.c $(SRC_FILES)
	$(CC) $(CFLAGS) -o benchmark $(DEFINES) $^

test: $(SRC_DIR)/test.c $(SRC_FILES)
	$(CC) $(CFLAGS) -o test $(DEFINES) $^

clean:
	rm -f benchmark
	rm -f test
