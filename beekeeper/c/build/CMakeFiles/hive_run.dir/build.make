# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/danilo/Documents/thesis/code/hive_engine/c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/danilo/Documents/thesis/code/hive_engine/c/build

# Include any dependencies generated for this target.
include CMakeFiles/hive_run.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hive_run.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hive_run.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hive_run.dir/flags.make

CMakeFiles/hive_run.dir/main.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/main.c.o: ../main.c
CMakeFiles/hive_run.dir/main.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/hive_run.dir/main.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/main.c.o -MF CMakeFiles/hive_run.dir/main.c.o.d -o CMakeFiles/hive_run.dir/main.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/main.c

CMakeFiles/hive_run.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/main.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/main.c > CMakeFiles/hive_run.dir/main.c.i

CMakeFiles/hive_run.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/main.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/main.c -o CMakeFiles/hive_run.dir/main.c.s

CMakeFiles/hive_run.dir/engine/moves.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/moves.c.o: ../engine/moves.c
CMakeFiles/hive_run.dir/engine/moves.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/hive_run.dir/engine/moves.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/moves.c.o -MF CMakeFiles/hive_run.dir/engine/moves.c.o.d -o CMakeFiles/hive_run.dir/engine/moves.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/moves.c

CMakeFiles/hive_run.dir/engine/moves.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/moves.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/moves.c > CMakeFiles/hive_run.dir/engine/moves.c.i

CMakeFiles/hive_run.dir/engine/moves.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/moves.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/moves.c -o CMakeFiles/hive_run.dir/engine/moves.c.s

CMakeFiles/hive_run.dir/engine/board.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/board.c.o: ../engine/board.c
CMakeFiles/hive_run.dir/engine/board.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/hive_run.dir/engine/board.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/board.c.o -MF CMakeFiles/hive_run.dir/engine/board.c.o.d -o CMakeFiles/hive_run.dir/engine/board.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/board.c

CMakeFiles/hive_run.dir/engine/board.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/board.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/board.c > CMakeFiles/hive_run.dir/engine/board.c.i

CMakeFiles/hive_run.dir/engine/board.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/board.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/board.c -o CMakeFiles/hive_run.dir/engine/board.c.s

CMakeFiles/hive_run.dir/pns/pn_tree.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/pns/pn_tree.c.o: ../pns/pn_tree.c
CMakeFiles/hive_run.dir/pns/pn_tree.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/hive_run.dir/pns/pn_tree.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/pns/pn_tree.c.o -MF CMakeFiles/hive_run.dir/pns/pn_tree.c.o.d -o CMakeFiles/hive_run.dir/pns/pn_tree.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/pns/pn_tree.c

CMakeFiles/hive_run.dir/pns/pn_tree.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/pns/pn_tree.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/pns/pn_tree.c > CMakeFiles/hive_run.dir/pns/pn_tree.c.i

CMakeFiles/hive_run.dir/pns/pn_tree.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/pns/pn_tree.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/pns/pn_tree.c -o CMakeFiles/hive_run.dir/pns/pn_tree.c.s

CMakeFiles/hive_run.dir/engine/list.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/list.c.o: ../engine/list.c
CMakeFiles/hive_run.dir/engine/list.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/hive_run.dir/engine/list.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/list.c.o -MF CMakeFiles/hive_run.dir/engine/list.c.o.d -o CMakeFiles/hive_run.dir/engine/list.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/list.c

CMakeFiles/hive_run.dir/engine/list.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/list.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/list.c > CMakeFiles/hive_run.dir/engine/list.c.i

CMakeFiles/hive_run.dir/engine/list.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/list.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/list.c -o CMakeFiles/hive_run.dir/engine/list.c.s

CMakeFiles/hive_run.dir/pns/pns.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/pns/pns.c.o: ../pns/pns.c
CMakeFiles/hive_run.dir/pns/pns.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/hive_run.dir/pns/pns.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/pns/pns.c.o -MF CMakeFiles/hive_run.dir/pns/pns.c.o.d -o CMakeFiles/hive_run.dir/pns/pns.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/pns/pns.c

CMakeFiles/hive_run.dir/pns/pns.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/pns/pns.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/pns/pns.c > CMakeFiles/hive_run.dir/pns/pns.c.i

CMakeFiles/hive_run.dir/pns/pns.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/pns/pns.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/pns/pns.c -o CMakeFiles/hive_run.dir/pns/pns.c.s

CMakeFiles/hive_run.dir/mm/mm.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/mm/mm.c.o: ../mm/mm.c
CMakeFiles/hive_run.dir/mm/mm.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/hive_run.dir/mm/mm.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/mm/mm.c.o -MF CMakeFiles/hive_run.dir/mm/mm.c.o.d -o CMakeFiles/hive_run.dir/mm/mm.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/mm/mm.c

CMakeFiles/hive_run.dir/mm/mm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/mm/mm.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/mm/mm.c > CMakeFiles/hive_run.dir/mm/mm.c.i

CMakeFiles/hive_run.dir/mm/mm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/mm/mm.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/mm/mm.c -o CMakeFiles/hive_run.dir/mm/mm.c.s

CMakeFiles/hive_run.dir/engine/node.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/node.c.o: ../engine/node.c
CMakeFiles/hive_run.dir/engine/node.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/hive_run.dir/engine/node.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/node.c.o -MF CMakeFiles/hive_run.dir/engine/node.c.o.d -o CMakeFiles/hive_run.dir/engine/node.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/node.c

CMakeFiles/hive_run.dir/engine/node.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/node.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/node.c > CMakeFiles/hive_run.dir/engine/node.c.i

CMakeFiles/hive_run.dir/engine/node.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/node.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/node.c -o CMakeFiles/hive_run.dir/engine/node.c.s

CMakeFiles/hive_run.dir/timing/timing.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/timing/timing.c.o: ../timing/timing.c
CMakeFiles/hive_run.dir/timing/timing.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object CMakeFiles/hive_run.dir/timing/timing.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/timing/timing.c.o -MF CMakeFiles/hive_run.dir/timing/timing.c.o.d -o CMakeFiles/hive_run.dir/timing/timing.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/timing/timing.c

CMakeFiles/hive_run.dir/timing/timing.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/timing/timing.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/timing/timing.c > CMakeFiles/hive_run.dir/timing/timing.c.i

CMakeFiles/hive_run.dir/timing/timing.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/timing/timing.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/timing/timing.c -o CMakeFiles/hive_run.dir/timing/timing.c.s

CMakeFiles/hive_run.dir/mm/evaluation.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/mm/evaluation.c.o: ../mm/evaluation.c
CMakeFiles/hive_run.dir/mm/evaluation.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building C object CMakeFiles/hive_run.dir/mm/evaluation.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/mm/evaluation.c.o -MF CMakeFiles/hive_run.dir/mm/evaluation.c.o.d -o CMakeFiles/hive_run.dir/mm/evaluation.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/mm/evaluation.c

CMakeFiles/hive_run.dir/mm/evaluation.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/mm/evaluation.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/mm/evaluation.c > CMakeFiles/hive_run.dir/mm/evaluation.c.i

CMakeFiles/hive_run.dir/mm/evaluation.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/mm/evaluation.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/mm/evaluation.c -o CMakeFiles/hive_run.dir/mm/evaluation.c.s

CMakeFiles/hive_run.dir/engine/tt.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/tt.c.o: ../engine/tt.c
CMakeFiles/hive_run.dir/engine/tt.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building C object CMakeFiles/hive_run.dir/engine/tt.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/tt.c.o -MF CMakeFiles/hive_run.dir/engine/tt.c.o.d -o CMakeFiles/hive_run.dir/engine/tt.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/tt.c

CMakeFiles/hive_run.dir/engine/tt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/tt.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/tt.c > CMakeFiles/hive_run.dir/engine/tt.c.i

CMakeFiles/hive_run.dir/engine/tt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/tt.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/tt.c -o CMakeFiles/hive_run.dir/engine/tt.c.s

CMakeFiles/hive_run.dir/mcts/mcts.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/mcts/mcts.c.o: ../mcts/mcts.c
CMakeFiles/hive_run.dir/mcts/mcts.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building C object CMakeFiles/hive_run.dir/mcts/mcts.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/mcts/mcts.c.o -MF CMakeFiles/hive_run.dir/mcts/mcts.c.o.d -o CMakeFiles/hive_run.dir/mcts/mcts.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/mcts/mcts.c

CMakeFiles/hive_run.dir/mcts/mcts.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/mcts/mcts.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/mcts/mcts.c > CMakeFiles/hive_run.dir/mcts/mcts.c.i

CMakeFiles/hive_run.dir/mcts/mcts.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/mcts/mcts.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/mcts/mcts.c -o CMakeFiles/hive_run.dir/mcts/mcts.c.s

CMakeFiles/hive_run.dir/engine/utils.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/engine/utils.c.o: ../engine/utils.c
CMakeFiles/hive_run.dir/engine/utils.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building C object CMakeFiles/hive_run.dir/engine/utils.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/engine/utils.c.o -MF CMakeFiles/hive_run.dir/engine/utils.c.o.d -o CMakeFiles/hive_run.dir/engine/utils.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/engine/utils.c

CMakeFiles/hive_run.dir/engine/utils.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/engine/utils.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/engine/utils.c > CMakeFiles/hive_run.dir/engine/utils.c.i

CMakeFiles/hive_run.dir/engine/utils.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/engine/utils.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/engine/utils.c -o CMakeFiles/hive_run.dir/engine/utils.c.s

CMakeFiles/hive_run.dir/puzzles.c.o: CMakeFiles/hive_run.dir/flags.make
CMakeFiles/hive_run.dir/puzzles.c.o: ../puzzles.c
CMakeFiles/hive_run.dir/puzzles.c.o: CMakeFiles/hive_run.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building C object CMakeFiles/hive_run.dir/puzzles.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/hive_run.dir/puzzles.c.o -MF CMakeFiles/hive_run.dir/puzzles.c.o.d -o CMakeFiles/hive_run.dir/puzzles.c.o -c /home/danilo/Documents/thesis/code/hive_engine/c/puzzles.c

CMakeFiles/hive_run.dir/puzzles.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/hive_run.dir/puzzles.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/danilo/Documents/thesis/code/hive_engine/c/puzzles.c > CMakeFiles/hive_run.dir/puzzles.c.i

CMakeFiles/hive_run.dir/puzzles.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/hive_run.dir/puzzles.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/danilo/Documents/thesis/code/hive_engine/c/puzzles.c -o CMakeFiles/hive_run.dir/puzzles.c.s

# Object files for target hive_run
hive_run_OBJECTS = \
"CMakeFiles/hive_run.dir/main.c.o" \
"CMakeFiles/hive_run.dir/engine/moves.c.o" \
"CMakeFiles/hive_run.dir/engine/board.c.o" \
"CMakeFiles/hive_run.dir/pns/pn_tree.c.o" \
"CMakeFiles/hive_run.dir/engine/list.c.o" \
"CMakeFiles/hive_run.dir/pns/pns.c.o" \
"CMakeFiles/hive_run.dir/mm/mm.c.o" \
"CMakeFiles/hive_run.dir/engine/node.c.o" \
"CMakeFiles/hive_run.dir/timing/timing.c.o" \
"CMakeFiles/hive_run.dir/mm/evaluation.c.o" \
"CMakeFiles/hive_run.dir/engine/tt.c.o" \
"CMakeFiles/hive_run.dir/mcts/mcts.c.o" \
"CMakeFiles/hive_run.dir/engine/utils.c.o" \
"CMakeFiles/hive_run.dir/puzzles.c.o"

# External object files for target hive_run
hive_run_EXTERNAL_OBJECTS =

hive_run: CMakeFiles/hive_run.dir/main.c.o
hive_run: CMakeFiles/hive_run.dir/engine/moves.c.o
hive_run: CMakeFiles/hive_run.dir/engine/board.c.o
hive_run: CMakeFiles/hive_run.dir/pns/pn_tree.c.o
hive_run: CMakeFiles/hive_run.dir/engine/list.c.o
hive_run: CMakeFiles/hive_run.dir/pns/pns.c.o
hive_run: CMakeFiles/hive_run.dir/mm/mm.c.o
hive_run: CMakeFiles/hive_run.dir/engine/node.c.o
hive_run: CMakeFiles/hive_run.dir/timing/timing.c.o
hive_run: CMakeFiles/hive_run.dir/mm/evaluation.c.o
hive_run: CMakeFiles/hive_run.dir/engine/tt.c.o
hive_run: CMakeFiles/hive_run.dir/mcts/mcts.c.o
hive_run: CMakeFiles/hive_run.dir/engine/utils.c.o
hive_run: CMakeFiles/hive_run.dir/puzzles.c.o
hive_run: CMakeFiles/hive_run.dir/build.make
hive_run: CMakeFiles/hive_run.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking C executable hive_run"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hive_run.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hive_run.dir/build: hive_run
.PHONY : CMakeFiles/hive_run.dir/build

CMakeFiles/hive_run.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hive_run.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hive_run.dir/clean

CMakeFiles/hive_run.dir/depend:
	cd /home/danilo/Documents/thesis/code/hive_engine/c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/danilo/Documents/thesis/code/hive_engine/c /home/danilo/Documents/thesis/code/hive_engine/c /home/danilo/Documents/thesis/code/hive_engine/c/build /home/danilo/Documents/thesis/code/hive_engine/c/build /home/danilo/Documents/thesis/code/hive_engine/c/build/CMakeFiles/hive_run.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hive_run.dir/depend

