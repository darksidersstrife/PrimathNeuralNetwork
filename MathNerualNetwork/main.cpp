#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

#define SCALE 1.0

double func(double x)
{
	return exp(x) / SCALE;
}

int main()
{
	NeuralNetwork nw(1);
	nw.AddLayer(100);
	nw.learningRate = 0.000001;
	nw.moment = 0.0;
	nw.initialize();
	double x = -10.0;
	double err1 = 1000000.0, err2 = 1000000.0;
	for (size_t j = 0; j < 300000 && err1 > 0.01 * 101.0; j++)
	{
		x = -10.0;
		vector<vector<double>> inputs, outputs;
		for (size_t i = 0; i <= 100; i++)
		{
			
			nw.TrainBatch(vector<std::vector<double>>(1, vector<double>(1, x / 10.0)), vector<std::vector<double>>(1, vector<double>(1, func(x))));
			inputs.push_back(std::vector<double>(1, x / 10.0));
			outputs.push_back(vector<double>(1, func(x)));
			x += 0.2;
		}
		//nw.TrainBatch(inputs, outputs);
		
		if (j % 1000 == 0)
		{
			err1 = 0.0;
			err2 = 0.0;
			x = -10.0;
			
			for (size_t i = 0; i <= 100; i++)
			{
				nw.calculate(std::vector<double>(1, x / 10.0));
				//cout << (1000 * nw.getOutput() - func(x)) << " ";
				double diff = (nw.getOutput() - func(x)) * SCALE;
				err1 += diff * diff;
				err2 += abs(diff);
				x += 0.2;
			}
			cout << endl << j / 1000 + 1 << " " << err1 / 101.0 << " " << err2 / 101.0;
		}
	}
	x = -10.0;
	err1 = 0.0, err2 = 0.0;
	ofstream gr("graph_exp4.csv");
	for (size_t i = 0; i <= 200; i++)
	{
		nw.calculate(std::vector<double>(1, x / 10.0));
		gr << x << ", " << nw.getOutput() << endl;
		err1 += (nw.getOutput() - func(x) * SCALE) * (nw.getOutput() - func(x) * SCALE);
		err2 += abs(nw.getOutput() - func(x) * SCALE);
		x += 0.1;
	}
	gr.close();
	cout << endl << err1 / 201.0 << " " << err2 / 201.0;
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