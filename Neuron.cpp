#include "Neuron.h"

neuron::neuron(vector<float> WeightVector, float BiasNumber) //constructor 1
{
	setWeights(WeightVector); //gives provided parameters to the class
	setBias(BiasNumber);
}

neuron::neuron(int WeightVectorSize) //constructor 2
{
	Weights.resize(WeightVectorSize);
	
	std::generate(Weights.begin(), Weights.end(), //generates random weights using algorithms and lambda function
		[&]() {
		return randomize(-1,1);
	});

	Bias = randomize(-1, 1);
}


neuron::~neuron() //destructor
{
}

neuron::neuron(const neuron &Neuron1) //copy constructor
{
	Weights = Neuron1.Weights;
	Bias = Neuron1.Bias;
	Output = Neuron1.Output;
}

neuron& neuron::operator = (const neuron& Neuron1) //assignment operator
{
	if (&Neuron1 != this)
	{
		Weights = Neuron1.Weights;
		Bias = Neuron1.Bias;
		Output = Neuron1.Output;
	}
	return *this;
}


float neuron::randomize(float Minimum, float Maximum)
{
	random_device RandomDevice; //Initializes random engine
	mt19937 Generator(RandomDevice()); //Mersenne Twister 19937 generator, rng
	uniform_real_distribution<float> Distribution(Minimum, Maximum); //uniform probability distribution between Minimum and Maximum
	return Distribution(Generator); //Generate random weights
}


void neuron::setWeights(vector<float> WeightVector)
{
	Weights = WeightVector; //sets the weights
	
}

void neuron::setBias(float BiasNumber)
{
	Bias = BiasNumber; //sets the bias
}
vector<float> neuron::getWeights()
{
	
	return Weights; //returns the weights
}

float neuron::getBias()
{
	return Bias; //returns the bias
}

const int neuron::getNumberOfInputs()
{
	return Weights.size(); //returns the nuber of inputs for the neurons
}

float* neuron::sigmoid(float* z)
{
	*z = 1 / (1 + exp(-*z)); // sigmoid function
	return z;
}

float* neuron::dsigmoid(float* z)
{
	float temp = *sigmoid(z); 
	*z = temp*(1 - temp); // derivative of the sigmoid
	return z;
}


float* neuron::activateFunc(vector<float*> input)
{
	Output = 0;
	
	vector<float> TInput(input.size()); //temporary vector to store transformed elements
	std::transform(input.begin(), input.end(), TInput.begin(), 
		[](float* &Element) {
		return *Element; //converts input, a vector of ptrs to TInput, a vector of floats
//last argument is a lambda function. It takes input of type floatpointer and sends it to code in {} to return a float
	});
	Output = std::inner_product(Weights.begin(), Weights.end(), TInput.begin(), Bias); //std algorithm to calculate the inner product, i.e. sum of products

	return &Output;
}

float* neuron::resultFunc(vector<float*> input) //calculates the output of a neuron
{
	Output = *sigmoid(activateFunc(input));
	return  &Output;
}

