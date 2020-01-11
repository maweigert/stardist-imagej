CC=gcc-9
CXX=g++-9

CFLAGS=-I./qhull_src/src/ -std=c++11 -O3 -fopenmp

QHULLCCSRC=qhull_src/src/libqhull_r/*.c
QHULLCXXSRC=qhull_src/src/libqhullcpp/*.cpp
SRC=libstardist3d*

BUILDDIR=build

all: test libstardist3d

qhull_cpp:
	mkdir -p $(BUILDDIR) && cd $(BUILDDIR) && $(CXX) -c -I../qhull_src/src/ ../qhull_src/src/libqhullcpp/*.cpp 

qhull_c:
	mkdir -p $(BUILDDIR) && cd $(BUILDDIR) && $(CC) -c -I../qhull_src/src/ ../qhull_src/src/libqhull_r/*.c

libstardist3d: libstardist3d.cpp libstardist3d.c libstardist3d.h utils.cpp $(BUILDDIR)/*.o
	$(CXX) -dynamiclib -std=c++11 -o libstardist3d.dylib utils.cpp libstardist3d.cpp libstardist3d.c $(BUILDDIR)/*.o $(CFLAGS) 

test: test.cpp libstardist3d
	$(CXX)  -o test test.cpp -L. -lstardist3d  $(CFLAGS) 

