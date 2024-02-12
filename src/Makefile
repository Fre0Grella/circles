# Makefile for the circles intersections program and his parallelized counterparts.
#
# Available targets:
#
# - make
#   builds the standard executable
#
# - make clean
#   remove all output files and executables
#
# - make movie
#   compile the executable circles.movie that writes a gnuplot
#   file at each iteration; the executable is run and the output
#   is processed to produce an animation circles.avi
#
# - make omp
#	builds the executable with the OpenMP directives
#
# - make omp-movie
#	compile the executable omp-circles.movie
#	and produce an animation
#
# - make cuda
#	builds the executable with the nvidia compiler	
#	using CUDA directives

# Last modified 2024-01-19 by Marco Galeri

CC=gcc
CUDACC=nvcc
EXE:=circles
OMPEXE:=omp-circles
CUDAEXE:=cuda-circles
CFLAGS=-std=c99 -Wall -Wpedantic
LDLIBS+=-lm

ALL: $(EXE)
$(EXE).movie: CFLAGS+=-DMOVIE
$(EXE).movie: $(EXE).c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

movie: $(EXE).movie
	./$(EXE).movie 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

omp: $(OMPEXE)
$(OMPEXE): CFLAGS+=-fopenmp
$(OMPEXE).movie: CFLAGS+=-DMOVIE
$(OMPEXE).movie: $(OMPEXE).c
	$(CC) $(CFLAGS) $(OMPEXE).c -o $(OMPEXE) $(LDLIBS)

omp-movie: $(OMPEXE).movie
	./$(OMPEXE) 200 500
	for f in *.gp; do echo "Processing $$f"; gnuplot "$$f"; done
	ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi

cuda:
	$(CUDACC) $(CUDAEXE).cu -o $(CUDAEXE) $(LDLIBS)

clean:
	\rm -f $(OMPEXE) $(CUDAEXE) $(EXE) $(EXE).movie *.o *~ *.gp *.png *.avi

