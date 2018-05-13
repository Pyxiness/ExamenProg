#include "Layer.h"


using std::cout;

float randomize(float Minimum, float Maximum)
{
	random_device RandomDevice; //Initializes random engine
	mt19937 Generator(RandomDevice()); //Mersenne Twister 19937 generator, rng
	uniform_real_distribution<float> Distribution(Minimum, Maximum); //uniform probability distribution between Minimum and Maximum
	return Distribution(Generator); //Generate random weights
}

int main() {
	float x = 1;


	int nneuron = 150; //number of neurons
	int ninputs = 728; //number of inputs per neuron
	vector<float> FInput(ninputs);

	std::generate(FInput.begin(), FInput.end(),
		[&]() {
		return 1;
	});

	//copy constructor and assignment operator test
	try {
		layer lay(nneuron, ninputs); //constructor2 for initialization
		vector<vector<float>> LW = lay.getWeights();
		vector<float> LB = lay.getBias();
		cout << "Weights: " << lay.getWeights()[0][0] << "," << lay.getWeights()[0][1] << " | " << lay.getWeights()[1][0] << "," << lay.getWeights()[1][1] << endl;
		cout << "Bias: " << lay.getBias()[0] << " | " << lay.getBias()[1] << endl;
		cout << "Result: " << lay(FInput)[0] << " | " << lay(FInput)[1] << endl;
		cout << "DSigmoid: " << lay.dsigmoid()[0] << " | " << lay.dsigmoid()[1] << endl;

		int loopsize = 100;
		for (int i = 0; i < loopsize; i++)
		{
			cout << "loop: " << i << endl;

			std::for_each(LW.begin(), LW.end(),
				[&](vector<float> &weight) {
				std::generate(weight.begin(), weight.end(),
					[&]() {
					return randomize(-1, 1);
				}
				);
				return weight;
			});
			std::generate(LB.begin(), LB.end(),
				[&]() {
				return randomize(-1, 1);
			});

			//update the parameters
			lay.setWeights(LW); //update the weights
			lay.setBias(LB); //update the bias
			cout << "Weights: " << lay.getWeights()[0][0] << "," << lay.getWeights()[0][1] << " | " << lay.getWeights()[1][0] << "," << lay.getWeights()[1][1] << endl;
			cout << "Bias: " << lay.getBias()[0] << " | " << lay.getBias()[1] << endl;
			cout << "Result: " << lay(FInput)[0] << " | " << lay(FInput)[1] << endl; //neuron output 
			cout << "DSigmoid: " << lay.dsigmoid()[0] << " | " << lay.dsigmoid()[1] << endl; //dsigmoid output


		}
	}
	catch (const invalid_argument& argerror)
	{
		cout << argerror.what() << endl;//outputs the error message
		return EXIT_FAILURE;

	}

}
