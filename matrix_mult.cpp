#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <future>

#define NUM_THREADS 2
#define NUM_TASKS 2
#define MATRIX_SIZE 1000
using namespace std::chrono;
// This simple program read 2 files as 2 matrices input and multiply them. 
// We conduct experiments with 1,2,4, and 8 threads and compare these results.
struct Matrix {
    float ** elements;
    void initialize_zero() {
        elements = new float * [MATRIX_SIZE];
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            elements[i] = new float[MATRIX_SIZE];
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                elements[i][j] = 0.0f;
            }
        }
    }

    void initialize() {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1e3, 1e3);
        
        elements = new float * [MATRIX_SIZE];
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            elements[i] = new float[MATRIX_SIZE];
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                elements[i][j] = dist(mt);
            }
        }
    } // initialize random

    void print() {
        std::cout << std::endl;
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            std::cout << "\t";

            for (int j = 0; j < MATRIX_SIZE; ++j) {
                std::cout << elements[i][j] << "\t";
            }
            std::cout << std::endl;
        }
    }
};

void multiply(Matrix& r, const Matrix& m1, const Matrix& m2);
void multiply_threading(Matrix& r, const int thread_number, const Matrix& m1, const Matrix& m2);
int multiply_async(Matrix& r, const int task_number, const Matrix& m1, const Matrix& m2);
void check_equal(Matrix& r1, Matrix& r2);

int main() {
    Matrix r, r1, r2, m1, m2;
    m1.initialize();
    m2.initialize();
    r.initialize_zero();
    r1.initialize_zero(); 
    r2.initialize_zero();
    // The naive approach for matrix multiplication
    auto start = high_resolution_clock::now();
    multiply(r, m1, m2);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds> (stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    // Matrix multiplication using multi-threading
    std::vector<std::thread> ths;
    start = high_resolution_clock::now(); // check the execution time

    for(int i = 0; i < NUM_THREADS; ++i) {
        ths.push_back(std::thread(&multiply_threading, std::ref(r1), i, std::ref(m1), std::ref(m2)));
    }

    for (auto& th: ths) {
        th.join();
    }

    stop = high_resolution_clock::now(); // end check time
    duration = duration_cast<microseconds> (stop - start);

    std::cout << "Time taken by multi threading: " << duration.count() << " microseconds" << std::endl;
    check_equal(r, r1);

    // Matrix multiplication using std::async
    std::vector<std::future<int>> tasks;
    start = high_resolution_clock::now(); // check the execution time
    for(int i = 0; i < NUM_TASKS; ++i) {
        tasks.push_back(std::async(multiply_async, std::ref(r2), i, std::ref(m1), std::ref(m2)));
    }

    for (auto& task: tasks) {
        task.get();
    }
    stop = high_resolution_clock::now(); // end check time
    duration = duration_cast<microseconds> (stop - start);

    std::cout << "Time taken by asynchronization: " << duration.count() << " microseconds" << std::endl;
    check_equal(r, r2);
}

void check_equal(Matrix& r1, Matrix& r2) {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            const int e1 = r1.elements[i][j];
            const int e2 = r2.elements[i][j];
            if (e1 - e2) {
                std::cout << "FALSE" << std::endl;
                return;
            } 
        }
    }
    std::cout << "TRUE" << std::endl;
}

void multiply(Matrix& r, const Matrix& m1, const Matrix& m2) {
    auto result = 0.0f;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            result = 0.0f;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                result += m1.elements[i][k] * m2.elements[k][j];
            }
            r.elements[i][j] = result;
        }
    }
}

void multiply_threading(Matrix& r, const int thread_number, const Matrix& m1, const Matrix& m2) {
    // Calculate workload
    const int n_elements = (MATRIX_SIZE * MATRIX_SIZE);
    const int n_operations = n_elements / NUM_THREADS;
    const int rest_operations = n_elements % NUM_THREADS;
    // identify the start index and end Index
    int start_index, end_index;
    
    if (thread_number == 0) {
        start_index = n_operations * thread_number;
        end_index = (n_operations * (thread_number + 1)) + rest_operations;
    } else {
        start_index = n_operations * thread_number + rest_operations;
        end_index = (n_operations * (thread_number + 1)) + rest_operations;
    }
    // find the real index in the two matrices
    float result = 0.0f;
    for (int pos = start_index; pos < end_index; ++pos) {
        int i = pos / MATRIX_SIZE;
        int j = pos % MATRIX_SIZE;
        result = 0.0f;
        for (int k = 0; k < MATRIX_SIZE; ++k) {
            result += m1.elements[i][k] * m2.elements[k][j];
        }
        r.elements[i][j] = result;
    }
}

int multiply_async(Matrix& r, const int task_number, const Matrix& m1, const Matrix& m2) {
    // Calculate workload
    const int n_elements = (MATRIX_SIZE * MATRIX_SIZE);
    const int n_operations = n_elements / NUM_THREADS;
    const int rest_operations = n_elements % NUM_THREADS;
    // identify the start index and end Index
    int start_index, end_index;
    
    if (task_number == 0) {
        start_index = n_operations * task_number;
        end_index = (n_operations * (task_number + 1)) + rest_operations;
    } else {
        start_index = n_operations * task_number + rest_operations;
        end_index = (n_operations * (task_number + 1)) + rest_operations;
    }
    // find the real index in the two matrices
    float result = 0.0f;
    for (int pos = start_index; pos < end_index; ++pos) {
        int i = pos / MATRIX_SIZE;
        int j = pos % MATRIX_SIZE;
        result = 0.0f;
        for (int k = 0; k < MATRIX_SIZE; ++k) {
            result += m1.elements[i][k] * m2.elements[k][j];
        }
        r.elements[i][j] = result;
    }
    return 0;
}