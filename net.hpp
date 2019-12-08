#pragma once

#include "cifar.hpp"
#include "matrix.hpp"

namespace net {
	cifar10::cifar10 data;

	int rd = 0;

	struct convolutionCell {
		std::vector<matrix::matrix> W;
		matrix::matrix dW;
		std::vector<double> b;

		convolutionCell() {}
		~convolutionCell() {}
	};

	struct samplingCell {
		matrix::matrix X;
		matrix::matrix dW;
		matrix::matrix Y;

		samplingCell() {}
		~samplingCell() {}
	};

	struct fullConnectCell {
		matrix::matrix W;
		double b;
		double delta;
		double Y;

		fullConnectCell() : b(0.1), delta(0.0), Y(0.0) {}
	};

	struct convolutionLayer {
		std::vector<convolutionCell> cells;

		convolutionLayer() {}
		~convolutionLayer() {}
	};

	struct samplingLayer {
		std::vector<samplingCell> cells;

		samplingLayer() {}
		~samplingLayer() {}
	};

	struct fullConnectLayer {
		matrix::matrix V;
		std::vector<fullConnectCell> cells;

		fullConnectLayer() {}
		~fullConnectLayer() {}
	};

	struct lenet5 {
		convolutionLayer C1;
		samplingLayer S2;
		convolutionLayer C3;
		samplingLayer S4;
		fullConnectLayer O5;

		lenet5() {
			C1.cells = std::vector<convolutionCell>(6);

			for (int i = 0; i < 6; ++i) {
				C1.cells[i].W = std::vector<matrix::matrix>(3);

				for (int channel = 0; channel < 3; ++channel) {
					C1.cells[i].W[channel] = matrix::norm(5, 5, 0.0, 0.1);
					C1.cells[i].b = std::vector<double>(3);
				}
				
				C1.cells[i].dW = matrix::zero(32, 32);
			}

			S2.cells = std::vector<samplingCell>(6);

			for (int i = 0; i < 6; ++i) {
				S2.cells[i].X = matrix::zero(32, 32);
				S2.cells[i].dW = matrix::zero(16, 16);
				S2.cells[i].Y = matrix::zero(16, 16);
			}

			C3.cells = std::vector<convolutionCell>(16);

			for (int i = 0; i < 16; ++i) {
				C3.cells[i].W = std::vector<matrix::matrix>(1);
				C3.cells[i].W[0] = matrix::norm(5, 5, 0.0, 0.1);
				C3.cells[i].b = std::vector<double>(1);
				C3.cells[i].dW = matrix::zero(16, 16);
			}

			S4.cells = std::vector<samplingCell>(16);

			for (int i = 0; i < 16; ++i) {
				S4.cells[i].X = matrix::zero(16, 16);
				S4.cells[i].dW = matrix::zero(8, 8);
				S4.cells[i].Y = matrix::zero(8, 8);
			}

			O5.V = matrix::zero(1, 1024);
			O5.cells = std::vector<fullConnectCell>(10);

			for (int i = 0; i < 10; ++i) {
				O5.cells[i].W = matrix::norm(1, 1024, 0.0, 0.1);
			}
		}
		~lenet5() {}
	};

	constexpr const unsigned int filter1[16][6] = {
		{1, 1, 1, 0, 0, 0}, {0, 1, 1, 1, 0, 0}, {0, 0, 1, 1, 1, 0}, {0, 0, 0, 1, 1, 1},
		{1, 0, 0, 0, 1, 1}, {1, 1, 0, 0, 0, 1}, {1, 1, 1, 1, 0, 0}, {0, 1, 1, 1, 1, 0},
		{0, 0, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1}, {1, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 1},
		{1, 1, 0, 1, 1, 0}, {0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 1}
	};

	constexpr const unsigned int filter2[6][16] = {
		{1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1},
		{1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1},
		{1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1},
		{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
		{0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1},
		{0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1}
	};

	// not full connect in C1 and C3

	inline void forwardC1(const cifar10::cifarNode& data, lenet5& net) {
		for (int i = 0; i < 6; ++i) {
			for (int channel = 0; channel < 3; ++channel) {
				matrix::matrix x_hat = matrix::padding(data.pixel[channel], 4);
				// we pad it to a 36 * 36 matrix to get the same size of a 32 * 32 output
				// what we call 'SAME'

				for (int r = 0; r < 32; ++r) {
					for (int c = 0; c < 32; ++c) {
						matrix::matrix ms = matrix::block(x_hat, r, c, 5);
						// matrix::matrix ms = matrix::block(data.pixel[channel], r, c, 5);

						double z = matrix::hadamard(ms, net.C1.cells[i].W[channel]);
						// double y = activate::sigmoid(z);

						net.S2.cells[i].X[r][c] += z + net.C1.cells[i].b[channel];
					}
				}
			}
		}

		for (int i = 0; i < 6; ++i) {
			for (int r = 0; r < 32; ++r) {
				for (int c = 0; c < 32; ++c) {
					net.S2.cells[i].X[r][c] = activate::sigmoid(net.S2.cells[i].X[r][c]);
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void forwardS2(lenet5& net) {
		for (int i = 0; i < 6; ++i) {
			for (int r = 0; r < 16; ++r) {
				for (int c = 0; c < 16; ++c) {
					double pmax = -1e30;

					pmax = activate::max(pmax, net.S2.cells[i].X[r << 1][c << 1]);
					pmax = activate::max(pmax, net.S2.cells[i].X[r << 1 | 1][c << 1]);
					pmax = activate::max(pmax, net.S2.cells[i].X[r << 1][c << 1 | 1]);
					pmax = activate::max(pmax, net.S2.cells[i].X[r << 1 | 1][c << 1 | 1]);

					net.S2.cells[i].Y[r][c] = pmax;
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}
	//max pooling

	inline void forwardC3(lenet5& net) {
		for (int i = 0; i < 16; ++i) {
			for (int r = 0; r < 16; ++r) {
				for (int c = 0; c < 16; ++c) {
					double Wx = 0.0;

					for (int j = 0; j < 6; ++j) {
						if (filter1[i][j] == 0) continue;

						matrix::matrix x_hat = matrix::padding(net.S2.cells[j].Y, 4);
						// we pad it to a 36 * 36 matrix to get the same size of a 32 * 32 output
						// what we call 'SAME'

						matrix::matrix ms = matrix::block(x_hat, r, c, 5);
						// matrix::matrix ms = matrix::block(net.S2.cells[j].Y, r, c, 5);

						Wx += matrix::hadamard(ms, net.C3.cells[i].W[0]);
					}

					double z = Wx + net.C3.cells[i].b[0];
					double y = activate::sigmoid(z);

					net.S4.cells[i].X[r][c] = y;
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void forwardS4(lenet5& net) {
		for (int i = 0; i < 16; ++i) {
			for (int r = 0; r < 8; ++r) {
				for (int c = 0; c < 8; ++c) {
					double pmax = -1e30;

					pmax = activate::max(pmax, net.S4.cells[i].X[r << 1][c << 1]);
					pmax = activate::max(pmax, net.S4.cells[i].X[r << 1 | 1][c << 1]);
					pmax = activate::max(pmax, net.S4.cells[i].X[r << 1][c << 1 | 1]);
					pmax = activate::max(pmax, net.S4.cells[i].X[r << 1 | 1][c << 1 | 1]);

					net.S4.cells[i].Y[r][c] = pmax;
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}
	//max pooling

	inline void forwardO5(lenet5& net) {
		for (int i = 0; i < 16; ++i) {
			for (int r = 0; r < 8; ++r) {
				for (int c = 0; c < 8; ++c) {
					net.O5.V[0][i * 8 * 8 + r * 8 + c] = net.S4.cells[i].Y[r][c];
				}
			}
		}
		// flatten

		for (int i = 0; i < 10; ++i) {
			double z = matrix::hadamard(net.O5.cells[i].W, net.O5.V) + net.O5.cells[i].b;
			double y = activate::sigmoid(z);

			net.O5.cells[i].Y = y;

			/*
			std::cerr << y;
			if (i == 9) std::cerr << std::endl;
			else std::cerr << " ";
			*/
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void fp(const cifar10::cifarNode& data, lenet5& net) {
		forwardC1(data, net);
		forwardS2(net);
		forwardC3(net);
		forwardS4(net);
		forwardO5(net);

		if (rd < 50000) std::cerr << "fp #" << (++rd) << " accomplished." << std::endl;
	}

	inline void backwardO5(const cifar10::cifarNode& data, lenet5& net) {

		for (int i = 0; i < 10; ++i) {
			double z = net.O5.cells[i].Y;

			net.O5.cells[i].delta = z - data.label[i];

			// std::cerr << "output: " << z << " logistic: " << net.O5.cells[i].delta << std::endl;
		}

		/*
		std::cerr << "label:" << std::endl;
		for (int i = 0; i < 10; ++i) {
			std::cerr << data.label[i];
			if (i == 9) std::cerr << std::endl;
			else std::cerr << " ";
		}
		*/

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}
	// softmax loss function

	inline void backwardS4(lenet5& net) {
		for (int i = 0; i < 16; ++i) {
			for (int r = 0; r < 8; ++r) {
				for (int c = 0; c < 8; ++c) {
					for (int j = 0; j < 10; ++j) {
						net.S4.cells[i].dW[r][c] += net.O5.cells[j].W[0][i * 8 * 8 + r * 8 + c] * net.O5.cells[j].delta;
					}
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void backwardC3(lenet5& net) {
		for (int i = 0; i < 16; ++i) {
			matrix::matrix delta = matrix::bpMaxPooling(16, net.S4.cells[i].dW, 2, net.S4.cells[i].Y, net.S4.cells[i].X);

			for (int r = 0; r < 16; ++r) {
				for (int c = 0; c < 16; ++c) {
					net.C3.cells[i].dW[r][c] += delta[r][c] * activate::dsigmoid(net.S4.cells[i].X[r][c]);
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void backwardS2(lenet5& net) {
		matrix::matrix tmp[16];

		for (int i = 0; i < 16; ++i) tmp[i] = matrix::zero(16, 16);

		for (int i = 0; i < 16; ++i) {
			for (int r = 0; r < 12; ++r) {
				for (int c = 0; c < 12; ++c) {
					matrix::blockAdd(tmp[i], r, c, matrix::valueMul(net.C3.cells[i].W[0], net.C3.cells[i].dW[r][c]));
				}
			}
		}

		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 16; ++j) {
				if (filter2[i][j] == 0) continue;

				net.S2.cells[i].dW = matrix::add(net.S2.cells[i].dW, tmp[j]);
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void backwardC1(lenet5& net) {
		for (int i = 0; i < 6; ++i) {
			matrix::matrix delta = matrix::bpMaxPooling(32, net.S2.cells[i].dW, 2, net.S2.cells[i].Y, net.S2.cells[i].X);

			for (int r = 0; r < 32; ++r) {
				for (int c = 0; c < 32; ++c) {
					net.C1.cells[i].dW[r][c] += delta[r][c] * activate::dsigmoid(net.S2.cells[i].X[r][c]);
				}
			}
		}

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void bp(const cifar10::cifarNode& data, lenet5& net) {
		backwardO5(data, net);
		backwardS4(net);
		backwardC3(net);
		backwardS2(net);
		backwardC1(net);

		if (rd <= 50000) std::cerr << "bp #" << rd << " accomplished." << std::endl;
	}

	inline void gradientDescendOptimizer(const cifar10::cifarNode& data, lenet5& net) {
		const double alpha = 0.01;

		/*
		std::cerr << "before gradient descend optimizer:" << std::endl;

		for (int i = 0; i < 6; ++i) {
			for (int channel = 0; channel < 3; ++channel) {
				matrix::print(net.C1.cells[i].W[channel]);
			}
		}

		for (int i = 0; i < 16; ++i) {
			matrix::print(net.C3.cells[i].W[0]);
		}

		for (int i = 0; i < 10; ++i) {
			matrix::print(net.O5.cells[i].W);
		}
		*/

		for (int i = 0; i < 6; ++i) {
			for (int channel = 0; channel < 3; ++channel) {
				matrix::matrix delta = matrix::zero(5, 5);
				matrix::matrix x_hat = matrix::padding(data.pixel[channel], 4);

				for (int r = 0; r < 32; ++r) {
					for (int c = 0; c < 32; ++c) {
						matrix::matrix ms = matrix::block(x_hat, r, c, 5);
						// matrix::matrix ms = matrix::block(data.pixel[channel], r, c, 5);

						delta = matrix::add(delta, matrix::valueMul(ms, net.C1.cells[i].dW[r][c]));
					}
				}

				/*
				std::cerr << "---" << std::endl;
				matrix::print(net.C1.cells[i].dW);
				std::cerr << "---" << std::endl;
				matrix::print(delta);
				std::cerr << "---" << std::endl;
				*/

				net.C1.cells[i].W[channel] = matrix::add(net.C1.cells[i].W[channel], matrix::valueMul(delta, -alpha));
				net.C1.cells[i].b[channel] -= alpha * matrix::sum(net.C1.cells[i].dW);
			}
		}

		for (int i = 0; i < 16; ++i) {
			matrix::matrix x = matrix::zero(16, 16);
			matrix::matrix delta = matrix::zero(5, 5);

			for (int j = 0; j < 6; ++j) {
				if (filter1[i][j] == 0) continue;

				x = matrix::add(x, net.S2.cells[j].Y);
			}

			matrix::matrix x_hat = matrix::padding(x, 4);

			for (int r = 0; r < 16; ++r) {
				for (int c = 0; c < 16; ++c) {
					matrix::matrix ms = matrix::block(x_hat, r, c, 5);
					// matrix::matrix ms = matrix::block(x, r, c, 5);

					delta = matrix::add(delta, matrix::valueMul(ms, net.C3.cells[i].dW[r][c]));
				}
			}

			net.C3.cells[i].W[0] = matrix::add(net.C3.cells[i].W[0], matrix::valueMul(delta, -alpha));
			net.C3.cells[i].b[0] -= alpha * matrix::sum(net.C3.cells[i].dW);
		}

		for (int i = 0; i < 10; ++i) {
			net.O5.cells[i].W = matrix::add(net.O5.cells[i].W, matrix::valueMul(net.O5.V, -alpha * net.O5.cells[i].delta));
			net.O5.cells[i].b -= alpha * net.O5.cells[i].delta;
		}

		/*
		std::cerr << "after gradient descend optimizer:" << std::endl;

		for (int i = 0; i < 6; ++i) {
			for (int channel = 0; channel < 3; ++channel) {
				matrix::print(net.C1.cells[i].W[channel]);
			}
		}

		for (int i = 0; i < 16; ++i) {
			matrix::print(net.C3.cells[i].W[0]);
		}

		for (int i = 0; i < 10; ++i) {
			matrix::print(net.O5.cells[i].W);
		}
		*/

		// std::cerr << "Passing function " << __FUNCTION__ << " line " << __LINE__ << std::endl;
	}

	inline void zero(lenet5& net) {
		for (int i = 0; i < 6; ++i) net.C1.cells[i].dW = matrix::zero(32, 32);
		for (int i = 0; i < 6; ++i) net.S2.cells[i].dW = matrix::zero(16, 16);
		for (int i = 0; i < 16; ++i) net.C3.cells[i].dW = matrix::zero(16, 16);
		for (int i = 0; i < 16; ++i) net.S4.cells[i].dW = matrix::zero(8, 8);
	}

	inline void train(const cifar10::cifar10& data, lenet5& net) {
		for (int i = 0; i < data.trainCases; ++i) {
			fp(data.trainSet[i], net);
			bp(data.trainSet[i], net);
			gradientDescendOptimizer(data.trainSet[i], net);
			zero(net);
		}
	}

	inline double recognize(const cifar10::cifar10& data, lenet5& net) {
		double ac = 0.0;

		for (int i = 0; i < data.testCases; ++i) {
			fp(data.testSet[i], net);

			double pmax = -1.0;
			int maxIndex = 0, stdIndex = 0;

			for (int j = 0; j < 10; ++j) {
				std::cerr << net.O5.cells[j].Y;

				if (j == 9) std::cerr << std::endl;
				else std::cerr << " ";

				if (pmax < net.O5.cells[j].Y) {
					pmax = net.O5.cells[j].Y;
					maxIndex = j;
				}
			}

			for (int j = 0; j < 10; ++j) {
				if (data.testSet[i].label[j] == 1.0) {
					stdIndex = j;
					break;
				}
			}

			if (maxIndex == stdIndex) ++ac;
			else std::cerr << "read " << cifar10::labels[maxIndex] << " except " << cifar10::labels[stdIndex] << std::endl;
		}

		return ac / data.testCases;
	}
}