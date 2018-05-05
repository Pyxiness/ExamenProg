#include "Layer.h"

layer::layer(vector<vector<float>>LayerWeights, vector<float> LayerBias, bool FirstLayerParam)
{
	if (LayerWeights.size() != LayerBias.size()) //argument check
	{
		throw std::invalid_argument("\n layer::layer1: dimension mismatch\n");
	}
	if (FirstLayerParam == true && LayerWeights.at(0).size() != 1)
	{
		throw std::invalid_argument("First layer can only have 1 input per neuron");
	}

	FirstLayer = FirstLayerParam; //sets the value for FirstLayer for further use in resultFunc and dsigmoid

	NumberOfNeurons = LayerWeights.size(); //#neurons = size of LayerWeights
	NumberOfInputs = LayerWeights.at(0).size(); //Because it is a priori known that every neuron takes an equal amount of inputs

	Neurons.reserve(NumberOfNeurons); //creates a vector of neurons using neuron/constructor2
	for (int i = 0; i < NumberOfNeurons; i++) //using iterators here is impractical, though we might fix it in future versions
	{
		Neurons.push_back(neuron(LayerWeights.at(i), LayerBias.at(i) ));
	}

	

}

layer::layer(int InitNumberOfNeurons, int InitNumberOfInputs, bool FirstLayerParam)
{
	//in every layer each neuron has the same amount of inputs and thus the same amount of weights
	if (InitNumberOfNeurons <= 0 || InitNumberOfInputs <= 0)
	{
		throw std::invalid_argument("\nlayer::layer2: invalid argument, argument must be of type int.\n");
	}
	if (FirstLayerParam == true && InitNumberOfInputs != 1)
	{
		throw std::invalid_argument("First layer can only have 1 input per neuron");
	}
	
	FirstLayer = FirstLayerParam;
	
	NumberOfNeurons = InitNumberOfNeurons; //sets parameters
	NumberOfInputs = InitNumberOfInputs;

	Neurons.reserve(NumberOfNeurons);
	for (int i = 0; i < NumberOfNeurons; i++) 
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
	FirstLayer = Layer1.FirstLayer;
	NumberOfNeurons = Layer1.NumberOfNeurons;
	NumberOfInputs = Layer1.NumberOfInputs;
}

layer& layer::operator = (const layer& Layer1)//assignment operator
{
	if (&Layer1 != this)
	{
		Neurons = Layer1.Neurons;
		FirstLayer = Layer1.FirstLayer;
		NumberOfNeurons = Layer1.NumberOfNeurons;
		NumberOfInputs = Layer1.NumberOfInputs;
	}
	return *this;
}

void layer::setWeights(vector<vector<float>> LayerWeights) //sets weights for every neuron
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

void layer::setBias(vector<float> LayerBias) //sets bias for every neuron
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

const int layer::getNumberOfNeurons() //the number of neurons in the layer
{
	return NumberOfNeurons;
}

vector<neuron> layer::getNeurons() //get information about the seperate neurons
{
	
	return Neurons;
}

vector<float*> layer::resultFunc(vector<float*> LayerInput) //calculates the output for each neuron in the layer
{
	vector<float*> LayerOutput(NumberOfNeurons); //temporary vector to store data
	//test if it's the first layer
	if (FirstLayer == true) //passes the i-th element of the input vector to the i-th neuron
	{
		if (LayerInput.size() != NumberOfNeurons)
		{
			throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
		}

		//Because every Neuron in the first layer has only one weight/input

		int count = 0; //indexer to access vector elements inside lambda function
		 //lamda function: [capture1,...] (type1,...) -> type {code} : executes the code inside {} and returns a value of type 'type'. () contains the type of input parameters for the code. [] contains the variables that the lambda can capture from outside its scope. In this case it can capture the index
		std::transform(Neurons.begin(), Neurons.end(),LayerOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.resultFunc({ LayerInput.at(count++) });
		});


	}
	//for other layers than the first
	else //passes the whole input vector to every neuron
	{
		if (LayerInput.size() != NumberOfInputs)
		{
			throw std::invalid_argument("\nlayer::resultFunc: dimension mismatch\n");
		}

		//Because every Neuron in the has the same input: all the outputs of the previous layer, so it is a priori known that every sub vector will have the same size
		
		std::transform(Neurons.begin(), Neurons.end(),LayerOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.resultFunc(LayerInput);
		});
	}
	return LayerOutput;
}

vector<float> layer::dsigmoid(vector<float*> Input)
{
	//same reasoning as in resultFunc
	vector<float> DSigmoidOutput(NumberOfNeurons);
	float* z;
	//test if it's the first layer 
	if (FirstLayer == true) //passes the i-th element of the input vector to the i-th neuron
	{
		if (Input.size() != NumberOfNeurons)
		{
			throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
		}

		int count = 0;
		std::transform(Neurons.begin(), Neurons.end(), DSigmoidOutput.begin(),
			[&](neuron &Neuron) {
			z = Neuron.activateFunc({ Input.at(count++) });
			return *Neuron.dsigmoid(z);
		});
	}
	//for other layers than the first
	else //passes the whole input vector to every neuron
	{
		if (Input.size() != NumberOfInputs)
		{
			throw std::invalid_argument("\nlayer::dsigmoid: dimension mismatch\n");
		}
		
		std::transform(Neurons.begin(), Neurons.end(),DSigmoidOutput.begin(),
			[&](neuron &Neuron) {
			z = Neuron.activateFunc(Input);
			return *Neuron.dsigmoid(z);
		});
	}
	return DSigmoidOutput;
}
