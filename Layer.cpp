
#include "Layer.h"

layer::layer(const vector<vector<float>>& LayerWeights, const vector<float>& LayerBias, bool FirstLayerParam)
{
	if (LayerWeights.size() != LayerBias.size())
	{
		throw invalid_argument("Both constructor inputs must be the same size");
	}
	if (FirstLayerParam == true && LayerWeights.at(0).size() != 1)
	{
		throw invalid_argument("First layer can only have 1 input per neuron");
	}

	FirstLayer = FirstLayerParam;
	setNumberOfNeurons(LayerBias.size());
	NumberOfInputs = LayerWeights.at(0).size(); //Because it is a priori known that every neuron takes an equal amount of inputs
	LayerOutput.resize(NumberOfNeurons);
	OutputPTR.resize(NumberOfNeurons);
	std::transform(LayerOutput.begin(), LayerOutput.end(), OutputPTR.begin(),
		[](float& num) {
		return &num;
	});
	Neurons.reserve(NumberOfNeurons);
	for (size_t i = 0; i < NumberOfNeurons; i++) //let's not use iterators here, shall we?
	{
		Neurons.push_back(neuron(LayerWeights.at(i), LayerBias.at(i)));
	}



}

layer::layer(const int& InitNumberOfNeurons, const int& InitNumberOfInputs, bool FirstLayerParam)
{
	//in every layer each neuron has the same amount of inputs and thus the same amount of weights
	if (InitNumberOfNeurons <= 0 || InitNumberOfInputs <= 0)
	{
		throw invalid_argument("Both constructor inputs must be a positive integer");
	}
	if (FirstLayerParam == true && InitNumberOfInputs != 1)
	{
		throw invalid_argument("First layer can only have 1 input per neuron");
	}
	//Neurons.assign(InitNumberOfNeurons, neuron(InitNumberOfInputs));
	FirstLayer = FirstLayerParam;
	setNumberOfNeurons(InitNumberOfNeurons);
	NumberOfInputs = InitNumberOfInputs;
	LayerOutput.resize(NumberOfNeurons);
	OutputPTR.resize(NumberOfNeurons);
	std::transform(LayerOutput.begin(), LayerOutput.end(), OutputPTR.begin(),
		[](float& num) {
		return &num;
	});
	Neurons.reserve(NumberOfNeurons);
	for (size_t i = 0; i < NumberOfNeurons; i++) //let's not use iterators here, shall we?
	{
		Neurons.push_back(neuron(NumberOfInputs));
	}


}

layer::~layer()
{

}

layer::layer(const layer &Layer1)
{
	Neurons = Layer1.Neurons;
	FirstLayer = Layer1.FirstLayer;
	NumberOfNeurons = Layer1.NumberOfNeurons;
	NumberOfInputs = Layer1.NumberOfInputs;
}

layer& layer::operator = (const layer& Layer1)
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

void layer::setWeights(const vector<vector<float>>& LayerWeights)
{
	if (LayerWeights.size() != NumberOfNeurons || LayerWeights.at(0).size() != NumberOfInputs)
	{
		throw invalid_argument("SetWeights():input has the wrong dimensions");
	}
	/*for (size_t i = 0; i < LayerWeights.size(); i++)
	{
	Neurons[i].setWeights( LayerWeights.at(i) );
	}*/

	int count = 0;
	std::for_each(Neurons.begin(), Neurons.end(),
		[&](neuron &Neuron) {
		Neuron.setWeights(LayerWeights.at(count++));
	});
}

void layer::setBias(const vector<float>& LayerBias)
{
	if (LayerBias.size() != NumberOfNeurons)
	{
		throw invalid_argument("setBias(): input has the wrong dimension");
	}
	/*for (size_t i = 0; i < LayerBias.size(); i++)
	{
	Neurons[i].setBias( LayerBias.at(i) );
	}*/

	int count = 0;
	std::for_each(Neurons.begin(), Neurons.end(),
		[&](neuron &Neuron) {
		Neuron.setBias(LayerBias.at(count++));
	});
}

vector<vector<float>> layer::getWeights()
{
	vector<vector<float>> LayerWeights(Neurons.size());
	std::transform(Neurons.begin(), Neurons.end(), LayerWeights.begin(),
		[](neuron &Neuron) {
		return Neuron.getWeights();
	});
	/*for (size_t i = 0; i < Neurons.size(); i++)
	{
	LayerWeights.at(i) = Neurons.at(i).getWeights();
	}*/
	return LayerWeights;
}

vector<float> layer::getBias()
{
	vector<float> LayerBias(Neurons.size());

	std::transform(Neurons.begin(), Neurons.end(), LayerBias.begin(),
		[](neuron &Neuron) {
		//cout << Neuron.getBias() << endl;
		return Neuron.getBias();
	});

	/*size_t it = 0;
	std::generate(LayerBias.begin(), LayerBias.end(),
	[&]() {
	return Neurons.at(it++).getBias();
	});*/
	/*for (size_t i = 0; i < Neurons.size(); i++)
	{
	LayerBias.at(i) = Neurons.at(i).getBias();
	}*/

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

const int layer::getNumberOfNeurons()
{
	return NumberOfNeurons;
}

vector<neuron> layer::getNeurons()
{

	return Neurons;
}

vector<float*> layer::resultFunc(const vector<float*>& LayerInput)
{
	//vector<float> LayerOutput(NumberOfNeurons);
	if (nullptr == LayerInput[0])
	{
		throw invalid_argument("resultFunc(): input doesn't point to anything");
	}

	if (FirstLayer == true)
	{
		if (LayerInput.size() != NumberOfNeurons)
		{
			throw invalid_argument("resultFunc(): input has the wrong dimension");
		}

		//Because every Neuron in the first layer has only one weight/input

		int count = 0;
		std::transform(Neurons.begin(), Neurons.end(), LayerOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.resultFunc({ LayerInput.at(count++) });
		});

		/*for (size_t i = 0; i < NumberOfNeurons; i++)
		{
		LayerOutput.at(i) = Neurons.at(i).resultFunc( { LayerInput.at(i) } );
		}*/

	}
	else
	{
		if (LayerInput.size() != NumberOfInputs)
		{
			throw invalid_argument("resultFunc(): input has the wrong dimension");
		}

		//Because every Neuron in the has the same input: all the outputs of the previous layer 
		/*for (size_t i = 0; i < NumberOfNeurons; i++)
		{
		LayerOutput.at(i) = Neurons.at(i).resultFunc(LayerInput);
		}*/

		std::transform(Neurons.begin(), Neurons.end(), LayerOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.resultFunc(LayerInput);
		});
	}
	return OutputPTR;
}

vector<float> layer::dsigmoid(const vector<float*>& Input)
{
	if (nullptr == Input[0])
	{
		throw invalid_argument("dsigmoid(): input doesn't point to anything");
	}

	//same reasoning as in resultFunc
	vector<float> DSigmoidOutput(NumberOfNeurons);
	if (FirstLayer == true)
	{
		if (Input.size() != NumberOfNeurons)
		{
			throw invalid_argument("dsigmoid(): input has the wrong dimension");
		}
		/*for (size_t i = 0; i < NumberOfNeurons; i++)
		{
		z = Neurons.at(i).activateFunc({ Input.at(i) });
		DSigmoidOutput.at(i) = *Neurons.at(i).dsigmoid(z);
		}*/
		std::transform(Neurons.begin(), Neurons.end(), DSigmoidOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.dsigmoid(Input);
		});
	}
	else
	{
		if (Input.size() != NumberOfInputs)
		{
			throw invalid_argument("dsigmoid(): input has the wrong dimension");
		}
		/*for (size_t i = 0; i < NumberOfNeurons; i++)
		{
		z = Neurons.at(i).activateFunc(Input);
		DSigmoidOutput.at(i) = *Neurons.at(i).dsigmoid(z);
		}*/
		//int count = 0;
		std::transform(Neurons.begin(), Neurons.end(), DSigmoidOutput.begin(),
			[&](neuron &Neuron) {
			return Neuron.dsigmoid(Input);
		});
	}
	return DSigmoidOutput;
}

vector<float> layer::dsigmoid()
{
	//same reasoning as in resultFunc
	vector<float> DSigmoidOutput(NumberOfNeurons);
	/*for (int i = 0; i < NumberOfNeurons; i++)
	{
	z = Neurons.at(i).activateFunc(Input);
	DSigmoidOutput.at(i) = *Neurons.at(i).dsigmoid(z);
	}*/
	//int count = 0;
	std::transform(Neurons.begin(), Neurons.end(), DSigmoidOutput.begin(),
		[&](neuron &Neuron) {
		return Neuron.dsigmoid();
	});
	return DSigmoidOutput;
}
