CXX := clang++

CXXFLAGS := -O3 -march=native -ffast-math -mfma -std=c++20 -Wall -Werror -Iincludes

SRCS := $(wildcard src/*.cpp)
TARGET := builds/test

.PHONY: all build run debug clean

all: build

build:
	@mkdir -p builds
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

run: build
	./$(TARGET)

debug: CXXFLAGS := -g -O0 -std=c++20 -Wall -Werror -Iincludes -fsanitize=address,undefined,leak
debug: build

clean:
	rm -rf build
	@echo "Cleaned up build directory!"