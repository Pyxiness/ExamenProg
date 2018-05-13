
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
	neuron(const vector<float>&, const float&); 			//Constructor1, weight and bias provided
	neuron(const int&); 						//Constructor2, no weight provided, start of the program
	~neuron(); 							//Destructor
	neuron(const neuron&); 						//copy constructor
	neuron& operator = (const neuron&); 				//assignment operator


	void setWeights(const vector<float>&); 				//Sets provided weights to vector Weights
	void setBias(const float&); 					//Sets provided bias to variable Bias

	vector<float> getWeights(); 					//get-function to access weights
	float getBias(); 						//get-function to access bias
	const int getNumberOfInputs(); 					//get-function to access #inputs = size of Weights

	void sigmoid(float&); 						//Sigmoid function
	float dsigmoid(); 						//Derivative Sigmoid function, output returned but not saved
	void activateFunc(const vector <float>&); 			//Activate function, calls sigmoid
	float resultFunc(const vector <float>&); 			//Calculates the neuron output, calls activateFunc
	float operator()(const vector<float>& NeuronInput) { return resultFunc(NeuronInput); } //Overloading ()
	float randomize(float, float); 					//rng function

protected:
	void setNumberOfInputs(const int&); 				//Sets number of inputs = number of weights
	vector<float> Weights;
	float Bias;
	size_t NumberOfInputs;
	float Output = 0;						//default set to zero
};

