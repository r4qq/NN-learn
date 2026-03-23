CXX := clang++

SRC_DIR := src
INC_DIR := includes
BUILD_ROOT := builds

DEBUG_DIR := $(BUILD_ROOT)/debug
RELEASE_DIR := $(BUILD_ROOT)/release

TARGET := test

BASE_FLAGS := -O3 -march=native -ffast-math -mfma -std=c++20 -Wall -Werror -I$(INC_DIR)
DEPFLAGS := -MMD -MP

SRCS := $(wildcard $(SRC_DIR)/*.cpp)

DEBUG_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/obj/%.o,$(SRCS))
RELEASE_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(RELEASE_DIR)/obj/%.o,$(SRCS))

.PHONY: all debug release run-debug run-release mem clean

all: release

debug: CXXFLAGS := $(BASE_FLAGS) -g -fsanitize=address,undefined,leak
debug: LDFLAGS := -fsanitize=address,undefined,leak
debug: $(DEBUG_DIR)/bin/$(TARGET)

$(DEBUG_DIR)/bin/$(TARGET): $(DEBUG_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Debug build complete: $@"

$(DEBUG_DIR)/obj/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

release: CXXFLAGS := $(BASE_FLAGS) -flto
release: LDFLAGS := -flto
release: $(RELEASE_DIR)/bin/$(TARGET)

$(RELEASE_DIR)/bin/$(TARGET): $(RELEASE_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Release build complete: $@"

$(RELEASE_DIR)/obj/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

run-debug: debug
	$(DEBUG_DIR)/bin/$(TARGET)

run-release: release
	$(RELEASE_DIR)/bin/$(TARGET)

mem: debug
	valgrind -s $(DEBUG_DIR)/bin/$(TARGET)

clean:
	rm -rf $(BUILD_ROOT)
	@echo "Clean complete"

-include $(DEBUG_OBJS:.o=.d)
-include $(RELEASE_OBJS:.o=.d)