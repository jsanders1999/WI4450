CXX=g++
#CXX_FLAGS=-O2 -g -fopenmp -std=c++17
CXX_FLAGS=-O3 -march=nmative -g -fopenmp -std=c++17

DEFS=-DNDEBUG

#default target (built when typing just "make")
default: run_tests.x main_cg_poisson.x main_cg_block.x main_benchmarks.x

# general rule to compile a C++ source file into an object file
%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

#define some dependencies on headers
operations.o: operations.hpp timer.hpp
cg_solver.o: cg_solver.hpp operations.hpp timer.hpp
cg_poisson.o: cg_solver.hpp operations.hpp timer.hpp
cg_block.o: cg_solver.hpp operations.hpp timer.hpp
gtest_mpi.o: gtest_mpi.hpp

TEST_SOURCES=test_operations.cpp test_cg_solver.cpp timer.o
MAIN_OBJ=main_cg_poisson.o cg_solver.o operations.o timer.o
MAIN_BLOCK=main_cg_block.o cg_solver.o operations.o timer.o
MAIN_BENCH=main_benchmarks.o cg_solver.o operations.o timer.o

run_tests.x: run_tests.cpp ${TEST_SOURCES} gtest_mpi.o operations.o cg_solver.o
	${CXX} ${CXX_FLAGS} ${DEFS} -o run_tests.x $^

main_cg_poisson.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_cg_poisson.x $^

main_cg_block.x: ${MAIN_BLOCK}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_cg_block.x $^

main_benchmarks.x: ${MAIN_BENCH}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main_benchmarks.x $^



test: run_tests.x
	./run_tests.x

clean:
	-rm *.o *.x

# phony targets are run regardless of dependencies being up-to-date
PHONY: clean, test

