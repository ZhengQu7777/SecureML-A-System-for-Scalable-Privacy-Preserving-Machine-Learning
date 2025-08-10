#ifndef SECML_UTIL_HPP
#define SECML_UTIL_HPP
#include "defines.hpp"
#include <iostream>
#include <vector>
#include <emp-tool/emp-tool.h>

void vector2d_to_RowMatrixXd(std::vector<std::vector<double>>& x, RowMatrixXd& X);
void vector_to_ColVectorXd(std::vector<double>& x, ColVectorXd& X);
void vector_to_RowVectorXi64(std::vector<uint64_t>& x, RowVectorXi64& X);
void vector2d_to_RowMatrixXi64(std::vector<std::vector<uint64_t>>& x, RowMatrixXi64& X);
void vector2d_to_ColMatrixXi64(std::vector<std::vector<uint64_t>>& x, ColMatrixXi64& X);
void vector_to_ColVectorXi64(std::vector<uint64_t>& x, ColVectorXi64& X);
void RowMatrixXi64_to_vector2d(RowMatrixXi64& X, std::vector<std::vector<uint64_t>>& x);
std::vector<uint64_t> ColVectorXi64_to_vector(ColVectorXi64 X);

void print128_num(emp::block var);
void print_binary(uint64_t int_);
void int_to_bool(uint64_t int_, bool* bool_);
int reverse_int(int i);


uint64_t extract_lo64(__m128i x);
uint64_t extract_hi64(__m128i x);

template<class Derived>
void send(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X){
    io->send_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
    return;
}

template<class Derived>
void recv(emp::NetIO* io, Eigen::PlainObjectBase<Derived>& X) {
    io->recv_data(X.data(), X.rows() * X.cols() * sizeof(uint64_t));
    return;
}

template<class Derived, class OtherDerived>
void scale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    Derived scaled_X = X * SCALING_FACTOR;
    x = scaled_X.template cast<uint64_t>();
    return;
}

template<class Derived, class OtherDerived>
void descale(Eigen::PlainObjectBase<Derived>& X, Eigen::PlainObjectBase<OtherDerived>& x) {
    x = (X.template cast<int64_t>()).template cast<double>();
    x /= SCALING_FACTOR;
    return;
}

template<class Derived>
void truncate(int i, uint64_t scaling_factor, Eigen::PlainObjectBase<Derived>& X) {
    if(i == 1) {
        X = -1 * X;
    }
    X /= SCALING_FACTOR;
    if(i == 1) {
        X = -1 * X;
    }
    return;
}

#endif