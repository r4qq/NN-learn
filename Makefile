CXX := clang++

CXXFLAGS := -O3 -march=native -ffast-math -mfma -std=c++20 -Wall -Werror -Iinclude -Isrc

EXAMPLE_DIR := examples
BUILD_DIR := builds

EXAMPLE_SRCS := $(shell find $(EXAMPLE_DIR) -name "*.cpp")

EXAMPLE_BINS := $(patsubst %.cpp, $(BUILD_DIR)/%, $(notdir $(EXAMPLE_SRCS)))

.PHONY: all clean help

all: $(EXAMPLE_BINS)

$(BUILD_DIR)/%:
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(shell find $(EXAMPLE_DIR) -name "$*.cpp") -o $@
	@echo "Built $@ successfully!"

clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned up builds directory!"

help:
	@echo "Detected source files:"
	@echo "$(EXAMPLE_SRCS)"
	@echo "Target binaries:"
	@echo "$(EXAMPLE_BINS)"