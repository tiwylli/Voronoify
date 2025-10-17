NVCC = nvcc
NVCC_ARCH ?= sm_86
NVCC_FLAGS = -O3 -arch=$(NVCC_ARCH) $(NVCC_FLAGS_EXTRA)
BIN_DIR = ../bin
SRC = voronoify_native.cu
TARGET = $(BIN_DIR)/voronoify_native

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
