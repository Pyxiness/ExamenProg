#pragma once
#include"Neuron.h"
#include<stdexcept>

class layer {
public:
	layer(const vector<vector<float>>&, const vector<float>&); 	//Constructor1, weights and biases provided
	layer(const int&, const int&); 					//Constructor2, random initialization of weights and biases
	~layer(); //default destructor


	void setWeights(const vector<vector<float>>&); 				//set-function for Weights across a layer
	void setBias(const vector<float>&); 					//set-function for bias across a layer

	vector<vector<float>> getWeights(); 					//get-function to provide access to weights
	vector<float> getBias(); 						//get-function to provide access to bias
	const int getNumberOfNeurons(); 					//get-function to provide access  to number of neurons

	vector<float>& resultFunc(const vector<float>&); 			//calculates the output vector of the layer.
	vector<float> dsigmoid();  						//calculates the derivative of sigmoid, returns value but doesn't save
	vector<float>& operator()(const vector<float> LayerInput) { return resultFunc(LayerInput); }
	layer(const layer&); 							//Copy constructor, don't use. Use constructor instead
	layer& operator = (const layer&); 					//Assignment constructor, don't use. Use constructor instead

	vector<float> LayerOutput;
	//vector<float*> OutputPTR;

protected:
	void setNumberOfNeurons(const int&);					//set-function for the number of neurons in a layer
	vector<neuron> Neurons;
	size_t NumberOfNeurons;
	size_t NumberOfInputs;


};
