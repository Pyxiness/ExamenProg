#include "Layer.h"
#include<chrono>

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
	float y = 0;
	auto start = std::chrono::system_clock::now();
	int NS = 150;
	int WS = 728;
	vector<float*> FInput(WS);
	/*for (int j = 0; j < WS; j++)
	{
		FInput[j] = &x;
	}*/
	std::generate(FInput.begin(),FInput.end(),
		[&]() {
		return &x;
	});

	try {
		layer pi(NS, WS);
		vector<vector<float>> LW = pi.getWeights();
		vector<float> LB = pi.getBias();
		cout << LB[0] << endl;
		cout << "Weights: " << pi.getWeights()[0][0] << "," << pi.getWeights()[0][1] << " | " << pi.getWeights()[1][0] << "," << pi.getWeights()[1][1] << endl;
		cout << "Bias: " << pi.getBias()[0] << " | " << pi.getBias()[1] << endl;
		cout << "Result: " << pi(FInput)[0] << " | " << pi(FInput)[1] << endl;
		cout << "DSigmoid: " << pi.dsigmoid(FInput)[0] << " | " << pi.dsigmoid(FInput)[1] << endl;

		int loopsize = 100;
		for (int i = 0; i < loopsize; i++)
		{
			cout << "loop: " << i << endl;
			/*for (int j = 0; j < NS; j++)
			{
				
				for (int k = 0; k < WS; k++)
				{
					LW[j][k] = randomize(-1, 1);
				}
				LB[j] = randomize(-1, 1);
			}*/
			std::for_each(LW.begin(),LW.end(),
				[&](vector<float> &weight) {
				std::generate(weight.begin(), weight.end(), 
					[&]() {
					return randomize(-1,1);
				}
					);
					return weight;
			});
			std::generate(LB.begin(), LB.end(),
				[&]() {
				return randomize(-1, 1);
			});
			pi.setWeights(LW);
			pi.setBias(LB);
			cout << "Weights: " << pi.getWeights()[0][0] << "," << pi.getWeights()[0][1] << " | "<< pi.getWeights()[1][0] << "," << pi.getWeights()[1][1] << endl;
			cout << "Bias: " << pi.getBias()[0]<< " | " << pi.getBias()[1] << endl;
			cout << "Result: " << pi(FInput)[0] << " | " << pi(FInput)[1] << endl;
			cout << "DSigmoid: " << pi.dsigmoid(FInput)[0] << " | " << pi.dsigmoid(FInput)[1] << endl;


		}
	}
catch (const invalid_argument& e)
{
	cout << e.what() << endl;

}
	auto end = std::chrono::system_clock::now();
	cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
	return 0;
}
