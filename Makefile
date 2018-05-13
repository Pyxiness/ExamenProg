#Ivan-Jago Coric
#Jana Christopher
#Stefan Koekkoek
#
#
#Makefile 

SHELL = /bin/bash
#Compiler
COMPILER = g++ -g -std=c++11 -Wall
#Macros not necessary


all:Neuron

Neuron: Neuron.o main.o Layer.o
	$(COMPILER) Neuron.o main.o Layer.o -o Neuron

main.o: main.cpp Layer.h Neuron.h
	$(COMPILER) main.cpp -c

Layer.o: Layer.cpp Layer.h
	$(COMPILER) -c Layer.cpp

Neuron.o: Neuron.cpp Neuron.h
	$(COMPILER) Neuron.cpp -c

clean : 
	rm *.o Neuron
	


