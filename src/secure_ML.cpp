#include "read_MNIST.hpp"
#include "linear_regression.hpp"

using namespace Eigen;
using namespace emp;
using namespace std;

//  <Eigen/Dense> -> defines.hpp -> util.hpp 

//  util.hpp -> read_MNIST.hpp
//  util.hpp & <emp-ot/emp-ot.h> -> setup_phase.hpp
//  util.hpp -> online_phase.hpp                                                  
                             
//  online_phase.hpp & setup_phase.hpp -> linear_regression.hpp
//  linear_regression.hpp -> secure_ML.cpp

IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ",",",","","","<<",";" );
int NUM_IMAGES = BATCH_SIZE;
int PARTY;

int main(int argc, char** argv) {
    int port, num_iters;
    string address;

    PARTY = atoi(argv[1]);
    port = atoi(argv[2]);
    num_iters = atoi(argv[3]);

    std::cout << "[main] PARTY=" << PARTY << ", port=" << port << ", num_iters=" << num_iters << std::endl;

    if(argc > 4) {
        address = atoi(argv[4]);
    }else {
        address = "127.0.0.1";
    }

    NUM_IMAGES *= num_iters;
    std::cout << "[main] NUM_IMAGES=" << NUM_IMAGES << std::endl;

    NetIO* io = new NetIO(PARTY == ALICE ? nullptr : address.c_str(), port);
    std::cout << "[main] NetIO initialized, address=" << address << std::endl;

    TrainingParams params;
    std::cout << "[main] TrainingParams created. n=" << params.n << ", d=" << params.d << std::endl;

    cout << "========" << "Training" << "========" << endl;

    vector<vector<uint64_t>> training_data;
    vector<uint64_t> training_labels;
    std::cout << "[main] Reading MNIST training data..." << std::endl;
    read_MNIST_data<uint64_t>(true, training_data, params.n, params.d);
    std::cout << "[main] Training data loaded. Size: " << training_data.size() << " x " << (training_data.empty() ? 0 : training_data[0].size()) << std::endl;
    RowMatrixXi64 X(params.n, params.d);
    vector2d_to_RowMatrixXi64(training_data, X);
    X *= SCALING_FACTOR;
    X /= 255;

    std::cout << "[main] Reading MNIST training labels..." << std::endl;
    read_MNIST_labels<uint64_t>(true, training_labels);
    std::cout << "[main] Training labels loaded. Size: " << training_labels.size() << std::endl;
    ColVectorXi64 Y(params.n);
    vector_to_ColVectorXi64(training_labels, Y);
    Y *= SCALING_FACTOR;
    Y /= 10;

    std::cout << "[main] Constructing LinearRegression..." << std::endl;
    LinearRegression linear_regression(X, Y, params, io);
    std::cout << "[main] LinearRegression constructed." << std::endl;

    cout << "=======" << endl;
    cout << "Testing" << endl;
    cout << "=======" << endl;

    vector<vector<double>> testing_data;
    int n_;
    std::cout << "[main] Reading MNIST testing data..." << std::endl;
    read_MNIST_data<double>(false, testing_data, n_, params.d);
    std::cout << "[main] Testing data loaded. Size: " << testing_data.size() << " x " << (testing_data.empty() ? 0 : testing_data[0].size()) << std::endl;
    RowMatrixXd testX (n_, params.d);
    vector2d_to_RowMatrixXd(testing_data, testX);
    testX /= 255.0;

    vector<double> testing_labels;
    std::cout << "[main] Reading MNIST testing labels..." << std::endl;
    read_MNIST_labels<double>(false, testing_labels);
    std::cout << "[main] Testing labels loaded. Size: " << testing_labels.size() << std::endl;

    ColVectorXd testY(n_);
    vector_to_ColVectorXd(testing_labels, testY);
    std::cout << "[main] Calling test_model..." << std::endl;
    linear_regression.test_model(testX, testY);
    std::cout << "[main] test_model finished." << std::endl;
    return 0;
}