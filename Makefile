# ──────────────── project-wide settings ────────────────
# compiler & flags
CXX      := g++
STD      := c++17
WARNINGS := -Wall -Wextra -Wpedantic
OPT      := -O3
INC      := -Iinclude
CXXFLAGS := -std=$(STD) $(OPT) $(WARNINGS) $(INC) -MMD -MP

# link libraries you actually need; remove or add to taste
LDLIBS   := -lfftw3 -llapacke -llapack -lblas

# output locations
SRCDIR   := src
OBJDIR   := build
BINDIR   := bin
TARGET   := critical_collap          # final executable

# ──────────────── derived variables (don’t edit) ────────────────
SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS  := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPS     := $(OBJECTS:.o=.d)           # auto-generated dependency files

# default rule ───────────────────────────────────────────────────
.PHONY: all
all: $(BINDIR)/$(TARGET)

# link step ──────────────────────────────────────────────────────
$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(OBJECTS) $(LDLIBS) -o $@

# compile step ───────────────────────────────────────────────────
# each .cpp → .o (with a matching .d dependency file)
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# auto-include generated *.d files (if any already exist)
-include $(DEPS)

# convenience directories ────────────────────────────────────────
$(OBJDIR) $(BINDIR):
	@mkdir -p $@

# ──────────────── extra targets ─────────────────────────────────
.PHONY: clean debug release

debug: CXXFLAGS := $(CXXFLAGS:-O3=-O0) -g -DDEBUG
debug: all

release:                                # explicit alias for optimised build
release: all

clean:
	@$(RM) -r $(OBJDIR) $(BINDIR)/$(TARGET) 2>/dev/null || true