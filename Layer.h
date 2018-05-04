#pragma once
#include"Neuron.h"
#include<stdexcept>

class layer {
public:
	layer(vector<vector<float>>, vector<float>,bool  FirstLayerParam = false); //Constructor1, weights and biases provided
	layer(int,int,bool  FirstLayerParam = false); //Constructor2, random initialization of weights and biases
	~layer(); //default destructor
	

	void setWeights(vector<vector<float>>); //set-function for Weights across a layer
	void setBias(vector<float>); //set-function for bias across a layer


	vector<vector<float>> getWeights(); //get-function to provide access to weights
	vector<float> getBias(); //get-function to provide access to bias
	const int getNumberOfNeurons(); //get-function to provide access  to number of neurons
	vector<neuron> getNeurons(); //get-function to provide access to individual neurons
	vector<float*> resultFunc(vector<float*>); // calculates the output vector of the layer.
	vector<float> dsigmoid(vector<float*>); //calculates the derivative of sigmoid
	vector<float*> operator()(vector<float*> LayerInput) { return resultFunc(LayerInput); }
	layer(const layer&); //Copy constructor, don't use. Use constructor instead
	layer& operator = (const layer&); //Assignment constructor, don't use. Use constructor instead

protected:
	vector<neuron> Neurons;
	bool FirstLayer; //Necessary to define how the input vector in resultFunc and dsigmoid is read
	int NumberOfNeurons;
	int NumberOfInputs;

	
};