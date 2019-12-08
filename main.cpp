#include <cstdlib>
#include "net.hpp"

cifar10::cifar10 data;
net::lenet5 inet;

int main() {
	cifar10::readCifar10(data);

	net::train(data, inet);

	double accuracy = net::recognize(data, inet);

	std::cout << "Accuracy: " << accuracy << std::endl;

	system("pause");

	return 0;
}