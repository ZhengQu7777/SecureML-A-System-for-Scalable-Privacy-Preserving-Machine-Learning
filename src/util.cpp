#include "util.hpp"

using namespace std;
using namespace Eigen;
using namespace emp;

void vector2d_to_RowMatrixXd(vector<vector<double>>& x, RowMatrixXd& X) {
    for(int i = 0; i < x.size(); i++) {
        X.row(i) << Map<RowVectorXd>(x[i].data(), x[i].size());
    }
    return;
}

void vector_to_ColVectorXd(vector<double>& x, ColVectorXd& X) {
    X << Map<ColVectorXd>(x.data(), x.size());
    return;
}

void vector_to_RowVectorXi64(vector<uint64_t>& x, RowMatrixXi64& X) {
    X << Map<RowVectorXi64>(x.data(), x.size());
    return;
}

void vector2d_to_RowMatrixXi64(vector<vector<uint64_t>>& x, RowMatrixXi64& X) {
    for(int i = 0; i < x.size(); i++) {
        X.row(i) << Map<RowVectorXi64>(x[i].data(), x[i].size()); 
    }
    return;
}

void vector2d_to_ColMatrixXi64(vector<vector<uint64_t>>& x, ColMatrixXi64& X) {
    for(int i = 0; i < x.size(); i++) {
        X.col(i) << Map<ColVectorXi64>(x[i].data(), x[i].size());
    }
    return;
}

void vector_to_ColVectorXi64(vector<uint64_t>& x, ColVectorXi64& X) {
    X << Map<ColVectorXi64>(x.data(), x.size());
    return;
}

void RowMatrixXi64_to_vector2d(RowMatrixXi64& X, vector<vector<uint64_t>>& x) {
    for(int i = 0; i < X.rows(); i++) {
        for(int j = 0; j < X.cols(); j++) {
            x[i][j] = X(i, j); 
        }
    }
    return;
}

vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X) {
    vector<uint64_t> x(X.data(), X.data() + X.rows());
    return x;
}

void print128_num(emp::block var){
    uint64_t* v64val = (uint64_t*) &var;
    printf("%016llx %016llx", v64val[0], v64val[1]);
}

void print_binary(uint64_t int_) {
    for(int i = 0; i < 64; i++) {
        cout << (int_ & 1);
        int_ >> 1;
    }
}

void int_to_bool( uint64_t int_, bool* bool_) {
    for(int i = 0; i < 64; i++) {
        bool_[i] = (int_ & 1);
        int_ >> 1;
    }
}

int reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

uint64_t extract_lo64(__m128i x) { // extract lower 64 bits of a block
    return (uint64_t) _mm_cvtsi128_si64(x);
}

uint64_t extract_hi64(__m128i x) { //extract higer 54 bits of a block
    uint64_t* v64val = (uint64_t*) &x;
    return v64val[1];
}


