#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace matrix {
	using rowVec = std::vector<double>;
	using matrix = std::vector<rowVec>;

	constexpr double mt19937_max = 4294967294;

	inline matrix zero(const int& r, const int& c) {
		matrix mat(r, rowVec(c, 0.0));

		return mat;
	}

	inline matrix norm(const int& r, const int& c, const double& mean, const double& stddev) {
		matrix mat(r, rowVec(c, 0.0));
		std::random_device e;
		std::mt19937 rng(e());
		std::normal_distribution<double> N(mean, stddev);

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				mat[i][j] = N(rng);
			}
		}

		return mat;
	}

	inline matrix valueMul(const matrix& lhs, const double& rhs) {
		int r = lhs.size(), c = lhs[0].size();
		matrix ret(r, rowVec(c, 0.0));

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				ret[i][j] = lhs[i][j] * rhs;
			}
		}

		return ret;
	}

	inline matrix add(const matrix& lhs, const matrix& rhs) {
		assert(lhs.size() == rhs.size() && lhs[0].size() == rhs[0].size());

		int r = lhs.size(), c = lhs[0].size();
		matrix ret(r, rowVec(c, 0.0));

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				ret[i][j] = lhs[i][j] + rhs[i][j];
			}
		}

		return ret;
	}

	inline double hadamard(const matrix& lhs, const matrix& rhs) {
		assert(lhs.size() == rhs.size() && lhs[0].size() == rhs[0].size());

		int r = lhs.size(), c = lhs[0].size();
		double ret = 0.0;

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				ret += lhs[i][j] * rhs[i][j];
			}
		}

		return ret;
	}

	inline double sum(const matrix& mat) {
		int r = mat.size(), c = mat[0].size();
		double ret = 0.0;

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				ret += mat[i][j];
			}
		}

		return ret;
	}

	inline matrix padding(const matrix& mat, int pad) {
		int r = mat.size(), c = mat[0].size();
		matrix ret(r + pad, rowVec(c + pad, 0.0));

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				ret[i + pad / 2][j + pad / 2] = mat[i][j];
			}
		}

		return ret;
	}

	inline matrix block(const matrix& mat, int leftmost, int topmost, int size) {
		matrix ret(size, rowVec(size, 0.0));

		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				ret[i][j] = mat[leftmost + i][topmost + j];
			}
		}

		return ret;
	}

	inline matrix bpMaxPooling(int pad, const matrix& mat, int size, const matrix& thisY, const matrix& prevX) {
		matrix ret(pad, rowVec(pad, 0.0));

		for (int i = 0; i < pad; ++i) {
			for (int j = 0; j < pad; ++j) {
				if (prevX[i][j] != thisY[i / size][j / size]) continue;

				ret[i][j] = mat[i / size][j / size];
			}
		}

		return ret;
	}

	inline void blockAdd(matrix& lhs, int leftmost, int topmost, const matrix& rhs) {
		int r = rhs.size(), c = rhs[0].size();

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				lhs[leftmost + i][topmost + j] += rhs[i][j];
			}
		}
	}

	inline void print(const matrix& mat) {
		int r = mat.size(), c = mat[0].size();

		for (int i = 0; i < r; ++i) {
			for (int j = 0; j < c; ++j) {
				std::cerr << mat[i][j];
				if (j == c - 1) std::cerr << std::endl;
				else std::cerr << " ";
			}
		}
	}
}

namespace activate {
	constexpr double max(const double& lhs, const double& rhs) {
		return lhs < rhs ? rhs : lhs;
	}

	constexpr double sigmoid(double z) {
		return 1.0 / (1.0 + exp(-z));
	}

	constexpr double dsigmoid(double y) {
		return y * (1.0 - y);
	}
}