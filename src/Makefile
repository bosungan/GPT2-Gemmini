# Compiler and Flags for pk
CC_PK = riscv64-unknown-linux-gnu-gcc

# Compiler and Flags for baremetal
CC_BAREMETAL = riscv64-unknown-elf-gcc

# Compilation Flags for pk
CFLAGS_PK = -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc

# Compilation Flags for baremetal
CFLAGS_BAREMETAL = -DBAREMETAL=1 -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -fno-tree-loop-distribute-patterns -march=rv64gc -Wa,-march=rv64gc

# Linking Flags for pk
LDFLAGS_PK = -lm -lgcc -static

# Linking Flags for baremetal
LDFLAGS_BAREMETAL = -lm -lgcc -nostdlib -nostartfiles -static -T ../include/riscv/test.ld

# Source and Object Files
SRCS = gpt2.c gpt2_params.c
OBJS_PK = $(SRCS:.c=-pk.o)
OBJS_BAREMETAL = $(SRCS:.c=-baremetal.o) syscalls.o crt.o

# Target Executables
TARGET_PK = gpt2-pk
TARGET_BAREMETAL = gpt2-baremetal

# Default Target
all: $(TARGET_PK) $(TARGET_BAREMETAL)

# PK Target
pk: $(TARGET_PK)

$(TARGET_PK): $(OBJS_PK)
	$(CC_PK) $(CFLAGS_PK) -o $@ $(OBJS_PK) $(LDFLAGS_PK)

# Compile each .c file to .o file for pk
%-pk.o: %.c
	$(CC_PK) $(CFLAGS_PK) -c $< -o $@

# Baremetal Target
baremetal: $(TARGET_BAREMETAL)

$(TARGET_BAREMETAL): $(OBJS_BAREMETAL)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -o $@ $(OBJS_BAREMETAL) $(LDFLAGS_BAREMETAL)

# Compile each .c file to .o file for baremetal
%-baremetal.o: %.c
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -c $< -o $@

# Compile each .S file to .o file for baremetal
%.o: %.S
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -c $< -o $@

# Compile additional .c and .S files for baremetal
syscalls.o: ../include/riscv/syscalls.c
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -c $< -o $@

crt.o: ../include/riscv/crt.S
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) -c $< -o $@

# Clean up build artifacts for pk
clean-pk:
	rm -f $(TARGET_PK) $(OBJS_PK)

# Clean up build artifacts for baremetal
clean-baremetal:
	rm -f $(TARGET_BAREMETAL) $(OBJS_BAREMETAL)

# Clean up all build artifacts
clean-all: clean-pk clean-baremetal

# Specify phony targets
.PHONY: all pk baremetal clean-pk clean-baremetal clean-all
