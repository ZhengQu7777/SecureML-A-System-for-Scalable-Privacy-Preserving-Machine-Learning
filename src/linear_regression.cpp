#include "linear_regression.hpp"
using namespace Eigen;
using namespace emp;
using namespace std;

void LinearRegression::train_model() {
    std::cout << "[train_model] party=" << party << ", t=" << t << ", n=" << n << std::endl;
    for(int i = 0; i < t; i ++) {
        int indexLo = (i * BATCH_SIZE) % n; // the starting position of current batch
        std::cout << "[train_model] Iter " << i << ", indexLo=" << indexLo << std::endl;
        online->train_batch(i, indexLo);
    }

    if(party == BOB) {
        send<ColVectorXi64>(io, online->wi);
    }else {
        recv<ColVectorXi64>(io, w);
    }

    if(party == ALICE) {
        send<ColVectorXi64>(io, online->wi);
    }else {
        recv<ColVectorXi64>(io, w);
    }

    w += online->wi;
    descale<ColVectorXi64, ColVectorXd>(w, w_d);
}


void LinearRegression::test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels) {
   
    ColVectorXd prediction;
    prediction = testing_data * w_d;
    prediction *= 10;
    int n_ = testing_labels.rows();
    prediction = round(prediction.array());

     // 调试：打印w_d、prediction、testing_data的统计信息
    std::cout << "[test_model][debug] w_d head: " << w_d.head(5).transpose() << std::endl;
    std::cout << "[test_model][debug] w_d tail: " << w_d.tail(5).transpose() << std::endl;
    std::cout << "[test_model][debug] w_d min: " << w_d.minCoeff() << ", max: " << w_d.maxCoeff() << std::endl;
    std::cout << "[test_model][debug] testing_data min: " << testing_data.minCoeff() << ", max: " << testing_data.maxCoeff() << std::endl;
    std::cout << "[test_model][debug] prediction (before round) min: " << prediction.minCoeff() << ", max: " << prediction.maxCoeff() << std::endl;
    std::cout << "[test_model] Testing started. test_data shape: " << testing_data.rows() << "x" << testing_data.cols() << std::endl;
    
    std::cout << "[test_model][debug] prediction (after round) min: " << prediction.minCoeff() << ", max: " << prediction.maxCoeff() << std::endl;

    // 打印前五个和后五个模型输出与标准输出
    std::cout << "[test_model] First 5 predictions vs labels:" << std::endl;
    for(int i = 0; i < std::min(5, n_); i++) {
        std::cout << "pred: " << prediction[i] << ", label: " << testing_labels[i] << std::endl;
    }
    std::cout << "[test_model] Last 5 predictions vs labels:" << std::endl;
    for(int i = std::max(0, n_-5); i < n_; i++) {
        std::cout << "pred: " << prediction[i] << ", label: " << testing_labels[i] << std::endl;
    }

    int num_correct = 0;
    for(int i = 0; i < n_; i++) {
        if(prediction[i] == testing_labels[i]) {
            num_correct++;
        }
    }
    double accuracy = num_correct / ((double) n_);
    std::cout << "[test_model] Accuracy on testing the trained model is " << accuracy * 100 << std::endl;
}
