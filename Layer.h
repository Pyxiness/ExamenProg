#pragma once
#include"Neuron.h"
#include<stdexcept>

class layer {
public:
	layer(const vector<vector<float>>&, const vector<float>&, bool  FirstLayerParam = false); //Constructor1, weights and biases provided
	layer(const int&, const int&, bool  FirstLayerParam = false); //Constructor2, random initialization of weights and biases
	~layer(); //default destructor


	void setWeights(const vector<vector<float>>&); //set-function for Weights across a layer
	void setBias(const vector<float>&); //set-function for bias across a layer


	vector<vector<float>> getWeights(); //get-function to provide access to weights
	vector<float> getBias(); //get-function to provide access to bias
	void setNumberOfNeurons(const int&);
	const int getNumberOfNeurons(); //get-function to provide access  to number of neurons
	vector<neuron> getNeurons(); //get-function to provide access to individual neurons
	vector<float*> resultFunc(const vector<float*>&); // calculates the output vector of the layer.
	vector<float> dsigmoid(const vector<float*>&); //calculates the derivative of sigmoid
	vector<float> dsigmoid();  //Same as above, but doesn't compute the neuron output again
	vector<float*> operator()(const vector<float*> LayerInput) { return resultFunc(LayerInput); }
	layer(const layer&); //Copy constructor, don't use. Use constructor instead
	layer& operator = (const layer&); //Assignment constructor, don't use. Use constructor instead

	vector<float> LayerOutput;
	vector<float*> OutputPTR;

protected:
	vector<neuron> Neurons;
	bool FirstLayer; //Necessary to define how the input vector in resultFunc and dsigmoid is read
	size_t NumberOfNeurons;
	size_t NumberOfInputs;


};
