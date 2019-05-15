#include <iostream>
#include "NeuralNetwork.h"

using namespace std;


double func(double x)
{
	return exp(x);
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double sigmoidDer(double x)
{
	return sigmoid(x) * 1 - sigmoid(x);
}

double tanh1(double x)
{
	return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

double tanhDer(double x)
{
	return (1 - tanh1(x) * tanh1(x));
}


int main()
{
	NeuralNetwork nw(1);
	//nw.AddLayer(10, sigmoid, sigmoidDer);
	nw.AddLayer(10, relu, reluDer);
	nw.learningRate = 0.0000000000000001;
	nw.initialize();
	double x = -10.0;
	for (size_t j = 0; j < 1000000; j++)
	{
		x = -10.0;
		vector<vector<double>> inputs, outputs;
		for (size_t i = 0; i <= 100; i++)
		{
			
			//nw.TrainBatch(vector<std::vector<double>>(1, vector<double>(1, x / 10.0)), vector<std::vector<double>>(1, vector<double>(1, func(x))));
			inputs.push_back(std::vector<double>(1, x / 10.0));
			outputs.push_back(vector<double>(1, func(x)));
			x += 0.2;
		}
		nw.TrainBatch(inputs, outputs);
		x = -10.0;
		double err1 = 0.0, err2 = 0.0;
		for (size_t i = 0; i <= 200; i++)
		{
			nw.calculate(std::vector<double>(1, x / 10.0));
			//cout << (1000 * nw.getOutput() - func(x)) << " ";
			err1 += (nw.getOutput() - func(x)) * (nw.getOutput() - func(x));
			err2 += abs(nw.getOutput() - func(x));
			x += 0.1;
		}
		cout << endl << err1 / 200.0 << endl << err2 / 200.0;
	}
	x = -10.0;
	double err1 = 0.0, err2 = 0.0;
	ofstream gr("graph.csv");
	for (size_t i = 0; i <= 200; i++)
	{
		nw.calculate(std::vector<double>(1, x / 10.0));
		gr << x << ", " << nw.getOutput() << endl;
		err1 += (nw.getOutput() - func(x)) * (nw.getOutput() - func(x));
		err2 += abs(nw.getOutput() - func(x));
		x += 0.1;
	}
	cout << endl << err1 / 200.0 << " " << err2 / 200.0;
	int b;
	cin >> b;
	return 0;
	/*NeuralNetwork nw(1);
	nw.AddLayer(100);
	nw.learningRate = 0.00001;
	nw.initialize();
	vector<vector<double>> inputs, outputs;
	inputs.push_back(vector<double>(1, 1));
	inputs.push_back(vector<double>(1, -1));
	outputs.push_back(vector<double>(1, func(9.2)));
	outputs.push_back(vector<double>(1, func(9.199)));
	for (size_t i = 0; i < 1000; i++)
	{
		nw.calculate(vector<double>(1, 1));
		cout << nw.getOutput() - func(9.2) << endl;
		nw.calculate(vector<double>(1, -1));
		cout << nw.getOutput() - func(9.199) << endl;
		//nw.TrainBatch(vector<std::vector<double>>(1, vector<double>(1, 1)), vector<std::vector<double>>(1, vector<double>(1, func(9.2))));
		//nw.TrainBatch(vector<std::vector<double>>(1, vector<double>(1, -1)), vector<std::vector<double>>(1, vector<double>(1, func(9.199))));
		nw.TrainBatch(inputs, outputs);
	}
	for (size_t i = 0; i < 100; i++)
	{
		nw.calculate(vector<double>(1, 1));
		cout << nw.getOutput() - func(9.2) << endl;
		nw.TrainBatch(vector<std::vector<double>>(1, vector<double>(1, 1)), vector<std::vector<double>>(1, vector<double>(1, func(9.2))));
	}
	cout << nw.getOutput() - func(9.2) << endl;
	int b;
	cin >> b;*/
}