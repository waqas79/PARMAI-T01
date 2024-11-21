#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm> // Include this for generate and other STL algorithms
#include <omp.h> // For parallel processing
using namespace std;

// Function to compute Sequential Prefix Sum
vector<int> sequential_prefix_sum(const vector<int>& input) {
    vector<int> result(input.size(), 0);
    result[0] = input[0];
    for (size_t i = 1; i < input.size(); i++) {
        result[i] = result[i - 1] + input[i];
    }
    return result;
}

// Function to compute Parallel Prefix Sum
vector<int> parallel_prefix_sum(const vector<int>& input) {
    vector<int> result(input.size(), 0);
    result[0] = input[0];

    #pragma omp parallel for
    for (size_t i = 1; i < input.size(); i++) {
        result[i] = result[i - 1] + input[i];
    }

    return result;
}

// Function to compute Scalar Product Sequentially
double scalar_product_sequential(const vector<double>& A, const vector<double>& B) {
    return inner_product(A.begin(), A.end(), B.begin(), 0.0);
}

// Function to compute Scalar Product in Parallel
double scalar_product_parallel(const vector<double>& A, const vector<double>& B, int num_threads) {
    double result = 0.0;
    #pragma omp parallel for reduction(+:result) num_threads(num_threads)
    for (size_t i = 0; i < A.size(); i++) {
        result += A[i] * B[i];
    }
    return result;
}

// Function to compute Matrix Multiplication
vector<vector<double>> matrix_multiplication(const vector<vector<double>>& A, const vector<vector<double>>& B, int num_threads) {
    size_t N = A.size();
    vector<vector<double>> C(N, vector<double>(N, 0));

    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            for (size_t k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main() {
    // Input Array for Prefix Sum
    vector<int> input_Array = {2, 4, 6, 8, 1, 3, 5, 7};

    // Measure time for Sequential Prefix Sum
    auto start_seq = chrono::high_resolution_clock::now();
    vector<int> seq_prefix_sum = sequential_prefix_sum(input_Array);
    auto end_seq = chrono::high_resolution_clock::now();
    double seq_time = chrono::duration<double>(end_seq - start_seq).count();

    // Measure time for Parallel Prefix Sum
    auto start_par = chrono::high_resolution_clock::now();
    vector<int> par_prefix_sum = parallel_prefix_sum(input_Array);
    auto end_par = chrono::high_resolution_clock::now();
    double par_time = chrono::duration<double>(end_par - start_par).count();

    cout << "Sequential Prefix Sum Time: " << seq_time << " seconds\n";
    cout << "Parallel Prefix Sum Time: " << par_time << " seconds\n";

    // Scalar Product
    vector<double> A(160), B(160);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 10);
    generate(A.begin(), A.end(), [&]() { return dis(gen); }); // Random generation
    generate(B.begin(), B.end(), [&]() { return dis(gen); }); // Random generation

    // Sequential Scalar Product
    auto start_scalar_seq = chrono::high_resolution_clock::now();
    double scalar_seq_result = scalar_product_sequential(A, B);
    auto end_scalar_seq = chrono::high_resolution_clock::now();
    double scalar_seq_time = chrono::duration<double>(end_scalar_seq - start_scalar_seq).count();

    // Parallel Scalar Product
    auto start_scalar_par = chrono::high_resolution_clock::now();
    double scalar_par_result = scalar_product_parallel(A, B, 8); // Using 8 threads
    auto end_scalar_par = chrono::high_resolution_clock::now();
    double scalar_par_time = chrono::duration<double>(end_scalar_par - start_scalar_par).count();

    cout << "Sequential Scalar Product Time: " << scalar_seq_time << " seconds\n";
    cout << "Parallel Scalar Product Time: " << scalar_par_time << " seconds\n";

    // Matrix Multiplication
    size_t N = 4;
    vector<vector<double>> matA(N, vector<double>(N)), matB(N, vector<double>(N));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            matA[i][j] = dis(gen);
            matB[i][j] = dis(gen);
        }
    }

    auto start_mat = chrono::high_resolution_clock::now();
    vector<vector<double>> matC = matrix_multiplication(matA, matB, 8); // Using 8 threads
    auto end_mat = chrono::high_resolution_clock::now();
    double mat_time = chrono::duration<double>(end_mat - start_mat).count();

    cout << "Matrix Multiplication Time: " << mat_time << " seconds\n";

    return 0;
}
