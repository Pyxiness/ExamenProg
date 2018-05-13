#include "Layer.h"

layer::layer(const vector<vector<float>>& LayerWeights, const vector<float>& LayerBias)
{
	if (LayerWeights.size() != LayerBias.size()) //argument check
	{
		throw std::invalid_argument("\n layer::layer1: dimension mismatch\n");
	}

	setNumberOfNeurons(LayerBias.size()); // sets the amouunt of neurons per layer
	NumberOfInputs = LayerWeights.at(0).size(); //Because it is a priori known that every neuron takes an equal amount of inputs
	LayerOutput.resize(NumberOfNeurons);
	/*OutputPTR.resize(NumberOfNeurons);
	std::transform(LayerOutput.begin(), LayerOutput.end(), OutputPTR.begin(),
		[](float& num) {
		return &num;
	});*/
	Neurons.reserve(NumberOfNeurons); //creates a vector of neurons using neuron/constructor2
	for (size_t i = 0; i < NumberOfNeurons; i++)
	{
		Neurons.push_back(neuron(LayerWeights.at(i), LayerBias.at(i)));
	}



}

layer::layer(const int& InitNumberOfNeurons, const int& InitNumberOfInputs)
{
	//in every layer each neuron has the same amount of inputs and thus the same amount of weights
	if (InitNumberOfNeurons <= 0 || InitNumberOfInputs <= 0)
	{
		throw std::invalid_argument("\nlayer::layer2: invalid argument, argument must be of type int.\n");
	}

	setNumberOfNeurons(InitNumberOfNeurons); //sets parameters
	NumberOfInputs = InitNumberOfInputs;
	LayerOutput.resize(NumberOfNeurons);
	/*OutputPTR.resize(NumberOfNeurons);
	std::transform(LayerOutput.begin(), LayerOutput.end(), OutputPTR.begin(),
		[](float& num) {
		return &num;
	});*/
	Neurons.reserve(NumberOfNeurons);
	for (size_t i = 0; i < NumberOfNeurons; i++)
	{
		Neurons.push_back(neuron(NumberOfInputs));
	}
}

layer::~layer() //destructor
{

}

layer::layer(const layer &Layer1) //copy constructor
{
	Neurons = Layer1.Neurons;
	NumberOfNeurons = Layer1.NumberOfNeurons;
	NumberOfInputs = Layer1.NumberOfInputs;
}

layer& layer::operator = (const layer& Layer1) //assignment operator
{
	if (&Layer1 != this)
	{
		Neurons = Layer1.Neurons;
		NumberOfNeurons = Layer1.NumberOfNeurons;
		NumberOfInputs = Layer1.NumberOfInputs;
	}
	return *this;
}

void layer::setWeights(const vector<vector<float>>& LayerWeights)  //sets weights for every neuron
{
	if (LayerWeights.size() != NumberOfNeurons || LayerWeights.at(0).size() != NumberOfInputs)
	{
		throw std::invalid_argument("\nlayer::setWeights: dimension mismatch\n");
	}

	int count = 0;
	std::for_each(Neurons.begin(), Neurons.end(),
		[&](neuron &Neuron) {
		Neuron.setWeights(LayerWeights.at(count++));
	});
}

void layer::setBias(const vector<float>& LayerBias) //sets bias for every neuron
{
	if (LayerBias.size() != NumberOfNeurons)
	{
		throw std::invalid_argument("\nlayer::setBias: dimension mismatch\n");
	}
	int count = 0;
	std::for_each(Neurons.begin(), Neurons.end(),
		[&](neuron &Neuron) {
		Neuron.setBias(LayerBias.at(count++));
	});
}

vector<vector<float>> layer::getWeights() //gets the weights for every neuron
{
	vector<vector<float>> LayerWeights(Neurons.size()); //temporary vector to store data
	std::transform(Neurons.begin(), Neurons.end(), LayerWeights.begin(),
		[](neuron &Neuron) {
		return Neuron.getWeights(); //std algorithm that transforms empty tmp to tmp filled with weight vectors
	});

	return LayerWeights;
}

vector<float> layer::getBias() //gets the bias for every neuron
{
	vector<float> LayerBias(Neurons.size());

	std::transform(Neurons.begin(), Neurons.end(), LayerBias.begin(),
		[](neuron &Neuron) {

		return Neuron.getBias();
	});


	return LayerBias;
}

void layer::setNumberOfNeurons(const int& InitNumberOfNeurons)
{
	if (0 >= InitNumberOfNeurons)
	{
		throw invalid_argument("setNumberOfNeurons(): input must be a positive int greater than 0");
	}
	NumberOfNeurons = InitNumberOfNeurons;
}

const int layer::getNumberOfNeurons() //the number of neurons in the layer
{
	return NumberOfNeurons;
}


vector<float>& layer::resultFunc(const vector<float>& LayerInput) //calculates the output for each neuron in the layer
{
	//test if it's the first layer
	if (LayerInput.size() != NumberOfInputs)
	{
		throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
	}

	//Because every Neuron in the has the same input: all the outputs of the previous layer,
	//so it is a priori known that every sub vector will have the same size

	std::transform(Neurons.begin(), Neurons.end(), LayerOutput.begin(),
		[&](neuron &Neuron) {
		return Neuron.resultFunc(LayerInput);
	});

	return LayerOutput;
}


vector<float> layer::dsigmoid()
{
	//same reasoning as in resultFunc
	vector<float> DSigmoidOutput(NumberOfNeurons);

	std::transform(Neurons.begin(), Neurons.end(), DSigmoidOutput.begin(),
		[&](neuron &Neuron) {
		return Neuron.dsigmoid();
	});
	return DSigmoidOutput;
}
