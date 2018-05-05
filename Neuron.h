#pragma once
#include<iostream>
#include<cmath>
#include<vector>
#include<random>
#include<algorithm>
#include<numeric>

using namespace std;


class neuron {
public:
	neuron(vector<float>, float); //Constructor1, weight provided
	neuron(int); //Constructor2, no weight provided, start of the program
	~neuron(); //Destructor
	neuron(const neuron&); //Copy constructor for Weights, Bias and Output
	neuron& operator = (const neuron&); //Assignment operator for neuron


	void setWeights(vector<float>); //Sets provided weights to vector Weights
	void setBias(float); //Sets provided bias to variable Bias

	vector<float> getWeights(); //get-function to access weights
	float getBias(); //get-function to access bias
	const int getNumberOfInputs(); //get-function to access #inputs = size of Weights
	float* sigmoid(float*); //Sigmoid function
	float* dsigmoid(float*); //Derivative Sigmoid function
	float* activateFunc(vector <float*>); //Activate function, calls sigmoid
	float* resultFunc(vector <float*>); //Calculates the neuron output, calls activateFunc
	float* operator()(vector<float*> NeuronInput) {return resultFunc(NeuronInput); } //Overloading ()
	float Output;
	
protected:
	vector<float> Weights;
	float Bias;
	float randomize(float, float); //rng function
};
