#pragma once
#include <vector>
#include <random>
#include <fstream>

double ActivationFunction(double x)
{
	//return x > 0.0 ? x : x / 100.0;
	return 1 / (1 + exp(-x));
}

double der(double x, double f)
{
	return f * (1 - f);
}

class NeuralNetwork
{
public:
	NeuralNetwork(size_t inputNumber, size_t outputNumber = 1)
	{
		lastLayer.neurons.resize(outputNumber);
		lastLayer.input.resize(inputNumber);
		for (size_t i = 0; i < outputNumber; i++)
		{
			lastLayer.neurons[i].inputWeights.resize(inputNumber + 1);
			lastLayer.neurons[i].previousWeights.resize(inputNumber + 1);
			lastLayer.neurons[i].corrections.resize(inputNumber + 1);
			lastLayer.neurons[i].qSum.resize(inputNumber + 1);
		}
		output.resize(outputNumber);
	}

	double getOutput()
	{
		return output[0];
	}

	void AddLayer(size_t size)
	{
		layers.push_back(Layer());

		layers[layers.size() - 1].input.resize(lastLayer.input.size());
		layers[layers.size() - 1].neurons.resize(size);
		for (size_t i = 0; i < size; i++)
		{
			layers[layers.size() - 1].neurons[i].inputWeights.resize(layers[layers.size() - 1].input.size() + 1);
			layers[layers.size() - 1].neurons[i].previousWeights.resize(layers[layers.size() - 1].input.size() + 1);
			layers[layers.size() - 1].neurons[i].corrections.resize(layers[layers.size() - 1].input.size() + 1);
			layers[layers.size() - 1].neurons[i].qSum.resize(layers[layers.size() - 1].input.size() + 1);
		}
		lastLayer.input.resize(size);
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			lastLayer.neurons[i].inputWeights.resize(size + 1);
			lastLayer.neurons[i].previousWeights.resize(size + 1);
			lastLayer.neurons[i].corrections.resize(size + 1);
			lastLayer.neurons[i].qSum.resize(size + 1);
		}
	}

	void initialize()
	{
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::normal_distribution<double> dis(0.0, 1);
		for (auto& layer : layers)
		{
			for (auto& neuron : layer.neurons)
			{
				for (size_t i = 0; i < neuron.inputWeights.size(); i++)
				{
					neuron.inputWeights[i] = dis(gen);
					neuron.previousWeights[i] = neuron.inputWeights[i];
				}
			}
		}
		for (auto& neuron : lastLayer.neurons)
		{
			for (size_t i = 0; i < neuron.inputWeights.size(); i++)
			{
				neuron.inputWeights[i] = dis(gen);
				neuron.previousWeights[i] = neuron.inputWeights[i];
			}
		}
	}

	void calculate(const std::vector<double>& input)
	{
		layers[0].input = input;
		for (size_t i = 0; i < layers.size() - 1; i++)
		{
			std::vector<Neuron>& neurons = layers[i].neurons;
			for (size_t j = 0; j < neurons.size(); j++)
			{
				layers[i + 1].input[j] = neurons[j](layers[i].input);
			}
		}
		Layer& last_hidden_layer = layers[layers.size() - 1];
		std::vector<Neuron>& neurons = last_hidden_layer.neurons;
		for (size_t j = 0; j < neurons.size(); j++)
		{
			lastLayer.input[j] = neurons[j](last_hidden_layer.input);
		}
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			output[i] = lastLayer.neurons[i].last(lastLayer.input);
		}
	}


	void TrainBatch(const std::vector< std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs)
	{
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			for (size_t j = 0; j < lastLayer.neurons[i].inputWeights.size() - 1; j++)
			{
				lastLayer.neurons[i].corrections[j] = 0;
			}
			lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1] = 0;
		}
		for (int k = layers.size() - 1; k >= 0; k--)
		{
			for (size_t i = 0; i < layers[k].neurons.size(); i++)
			{
				for (size_t j = 0; j < layers[k].neurons[i].inputWeights.size() - 1; j++)
				{
					layers[k].neurons[i].corrections[j] = 0;
				}
				layers[k].neurons[i].corrections[layers[k].neurons[i].inputWeights.size() - 1] = 0;
			}
		}
		for (size_t i = 0; i < inputs.size(); i++)
		{
			TrainOne(inputs[i], outputs[i]);
			//TrainOne(inputs[i], outputs[i]);
		}
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			for (size_t j = 0; j < lastLayer.neurons[i].inputWeights.size() - 1; j++)
			{
				lastLayer.neurons[i].previousWeights = lastLayer.neurons[i].inputWeights;
				lastLayer.neurons[i].inputWeights[j] += (lastLayer.neurons[i].corrections[j]);
			}
			lastLayer.neurons[i].previousWeights[lastLayer.neurons[i].inputWeights.size() - 1] = lastLayer.neurons[i].inputWeights[lastLayer.neurons[i].inputWeights.size() - 1];
			lastLayer.neurons[i].inputWeights[lastLayer.neurons[i].inputWeights.size() - 1] += (lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1]);
		}
		for (int k = layers.size() - 1; k >= 0; k--)
		{
			for (size_t i = 0; i < layers[k].neurons.size(); i++)
			{
				for (size_t j = 0; j < layers[k].neurons[i].inputWeights.size() - 1; j++)
				{
					layers[k].neurons[i].previousWeights[j] = layers[k].neurons[i].inputWeights[j];
					layers[k].neurons[i].inputWeights[j] += (layers[k].neurons[i].corrections[j]);
				}
				layers[k].neurons[i].previousWeights[layers[k].neurons[i].inputWeights.size() - 1] = layers[k].neurons[i].inputWeights[layers[k].neurons[i].inputWeights.size() - 1];
				layers[k].neurons[i].inputWeights[layers[k].neurons[i].inputWeights.size() - 1] += (layers[k].neurons[i].corrections[layers[k].neurons[i].inputWeights.size() - 1]);
			}
		}
	}

	void TrainOne(const std::vector<double>& input, const std::vector<double>& output)
	{
		log << input[0] << " output neuron" << std::endl;
		for (auto& weight : lastLayer.neurons[0].inputWeights)
		{
			log << weight << " ";
		}
		log << std::endl;
		calculate(input);
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			lastLayer.neurons[i].error = output[i] - this->output[i];
			lastLayer.neurons[i].delta = lastLayer.neurons[i].error;
		}
		
		std::vector<Neuron>& neurons = layers[layers.size() - 1].neurons;
		for (size_t i = 0; i < neurons.size(); i++)
		{
			neurons[i].error = 0;
			std::vector<Neuron>& next_layer_neurons = lastLayer.neurons;
			for (size_t j = 0; j < next_layer_neurons.size(); j++)
			{
				neurons[i].error += next_layer_neurons[j].delta * next_layer_neurons[j].inputWeights[i];
			}
			neurons[i].delta = neurons[i].error * der(neurons[i].sum, neurons[i].func);
		}
		for (int k = layers.size() - 2; k >= 0; k--)
		{
			std::vector<Neuron>& neurons = layers[k].neurons;
			for (size_t i = 0; i < neurons.size(); i++)
			{
				neurons[i].error = 0;
				std::vector<Neuron>& next_layer_neurons = layers[k + 1].neurons;
				for (size_t j = 0; j < next_layer_neurons.size(); j++)
				{
					neurons[i].error += next_layer_neurons[j].delta * next_layer_neurons[j].inputWeights[i];
				}
				neurons[i].delta = neurons[i].error * der(neurons[i].sum, neurons[i].func);
			}
		}
		for (size_t i = 0; i < lastLayer.neurons.size(); i++)
		{
			for (size_t j = 0; j < lastLayer.neurons[i].inputWeights.size() - 1; j++)
			{
				
				lastLayer.neurons[i].corrections[j] += 2 * learningRate * lastLayer.neurons[i].delta * lastLayer.input[j] + moment * (lastLayer.neurons[i].inputWeights[j] - lastLayer.neurons[i].previousWeights[j]);
				lastLayer.neurons[i].qSum[j] += lastLayer.neurons[i].corrections[j] * lastLayer.neurons[i].corrections[j];
				log << lastLayer.neurons[i].corrections[j] << " " << lastLayer.input[j] << " ";
			}
			lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1] += 2 * learningRate * lastLayer.neurons[i].delta + moment * (lastLayer.neurons[i].inputWeights[lastLayer.neurons[i].inputWeights.size() - 1] - lastLayer.neurons[i].previousWeights[lastLayer.neurons[i].inputWeights.size() - 1]);
			lastLayer.neurons[i].qSum[lastLayer.neurons[i].inputWeights.size() - 1] += lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1] * lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1];
			log << lastLayer.neurons[i].corrections[lastLayer.neurons[i].inputWeights.size() - 1] << " " << 1 << " ";
		}
		for (int k = layers.size() - 1; k >= 0; k--)
		{
			for (size_t i = 0; i < layers[k].neurons.size(); i++)
			{
				for (size_t j = 0; j < layers[k].neurons[i].inputWeights.size() - 1; j++)
				{
					layers[k].neurons[i].corrections[j] += 2 * learningRate * layers[k].neurons[i].delta * layers[k].input[j] + moment * (layers[k].neurons[i].inputWeights[j] - layers[k].neurons[i].previousWeights[j]);
					layers[k].neurons[i].qSum[j] += layers[k].neurons[i].corrections[j] * layers[k].neurons[i].corrections[j];
					
				}
				layers[k].neurons[i].corrections[layers[k].neurons[i].inputWeights.size() - 1] += + 2 * learningRate * layers[k].neurons[i].delta + moment * (layers[k].neurons[i].inputWeights[layers[k].neurons[i].inputWeights.size() - 1] - layers[k].neurons[i].previousWeights[layers[k].neurons[i].inputWeights.size() - 1]);
				layers[k].neurons[i].qSum[layers[k].neurons[i].inputWeights.size() - 1] += layers[k].neurons[i].corrections[layers[k].neurons[i].inputWeights.size() - 1] * layers[k].neurons[i].corrections[layers[k].neurons[i].inputWeights.size() - 1];
			}
		}
		log << std::endl;
		log << "calculated - " << this->output[0] << std::endl << "error - " << lastLayer.neurons[0].delta;
		log << std::endl << std::endl << std::endl;
	}

	double learningRate;
	double moment;
private:
	struct Neuron
	{
		double operator()(std::vector<double>& input)
		{
			double res = 0.0;
			for (size_t i = 0; i < input.size(); i++)
			{
				res += inputWeights[i] * input[i];
			}
			sum = res + inputWeights[inputWeights.size() - 1];
			func = ActivationFunction(sum);
			return func;
		}
		double last(std::vector<double>& input)
		{
			double res = 0.0;
			for (size_t i = 0; i < input.size(); i++)
			{
				res += inputWeights[i] * input[i];
			}
			sum = res + inputWeights[inputWeights.size() - 1];
			func = sum;
			return func;
		}
		std::vector<double> inputWeights;
		std::vector<double> previousWeights;
		std::vector<double> corrections;
		std::vector<double> qSum;
		double sum;
		double func;
		double error;
		double delta;
	};
	struct Layer
	{
		std::vector<Neuron> neurons;
		std::vector<double> input;
	};
	std::vector<Layer> layers;
	Layer lastLayer;
	std::vector<double> output;
	std::ofstream log/* = std::move(std::ofstream("weights_log.txt"))*/;
};