#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "matrix.hpp"

namespace cifar10 {
	constexpr const int n_channels = 3;
	constexpr const int n_pixels = 32;
	constexpr const int n_bytes = 3073;
	constexpr const int batch_size = 10000;
	const std::string prefix = "cifar-10-batches-bin/data_batch_";
	const std::string suffix = ".bin";
	const std::string test = "cifar-10-batches-bin/test_batch.bin";
	const std::vector<std::string> labels = {
		"airplane",
		"automobile",
		"bird",
		"cat",
		"deer",
		"dog",
		"frog",
		"horse",
		"ship",
		"truck"
	};

	struct cifarNode {
		matrix::matrix pixel[n_channels];
		matrix::rowVec label;

		cifarNode() {
			for (int i = 0; i < n_channels; ++i) {
				pixel[i] = matrix::matrix(n_pixels, matrix::rowVec(n_pixels, 0.0));
			}

			label = matrix::rowVec(10, 0.0);
		}
		~cifarNode() {}
	};

	struct cifar10 {
		int trainCases;
		int testCases;
		std::vector<cifarNode> trainSet;
		std::vector<cifarNode> testSet;

		cifar10() {}
		~cifar10() {}
	};

	inline void readBatch(const std::string& filename, std::vector<cifarNode>& data) {
		std::ifstream in(filename, std::ifstream::binary);

		if (!in.is_open()) {
			std::cerr << "open file " << filename << " failed." << std::endl;
			return;
		}

		char* mempool = new char[n_bytes];

		for (int i = 0; i < batch_size; ++i) {
			cifarNode newNode;

			in.read(mempool, n_bytes);

			newNode.label[mempool[0]] = 1.0;

			for (int channel = 0; channel < n_channels; ++channel) {
				for (int r = 0; r < n_pixels; ++r) {
					for (int c = 0; c < n_pixels; ++c) {
						int index = channel * (n_pixels * n_pixels) + r * n_pixels + c;

						double x = (double)(unsigned char)mempool[index + 1];
						newNode.pixel[channel][r][c] = x / 255.0;
					}
				}
			}

			data.emplace_back(newNode);
		}

		delete[] mempool;
	}

	inline void readTrain(cifar10& data) {
		data.trainCases = 5 * batch_size;

		for (char mid = '1'; mid <= '5'; ++mid) {
			std::string filename = prefix + mid + suffix;

			readBatch(filename, data.trainSet);

			std::random_shuffle(data.trainSet.begin(), data.trainSet.end());
		}
	}

	inline void readTest(cifar10& data) {
		data.testCases = batch_size;

		readBatch(test, data.testSet);
	}

	inline void readCifar10(cifar10& data) {
		readTrain(data);
		readTest(data);
	}
}