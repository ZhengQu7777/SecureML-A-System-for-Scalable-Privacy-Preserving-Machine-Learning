#include "online_phase.hpp"

using namespace Eigen;
using namespace emp;
using namespace std;

void OnlinePhase::initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi){
    this->Xi = Xi;
    this->Yi = Yi;

    for(int i = 0; i < d; i++) {
        wi(i) = 0;
    }

    Ui = triples->Ai;
    Ei = Xi - Ui;
    Vi = triples->Bi;
    Vi_ = triples->Bi_;
    Zi = triples->Ci;
    Zi_ = triples->Ci_;

    if(party == ALICE) {
        send<RowMatrixXi64>(io, Ei);
    }else {
        recv<RowMatrixXi64>(io, E);
    }

    if(party == BOB) {
        send<RowMatrixXi64>(io, Ei);
    }else {
        recv<RowMatrixXi64>(io, E);
    }

    E += Ei;
}

void OnlinePhase::train_batch(int iter, int indexLo) {
    
    std::cout << "[train_batch] iter=" << iter << ", indexLo=" << indexLo << ", party=" << party << std::endl;
    RowMatrixXi64 X = Xi.block(indexLo, 0, BATCH_SIZE, d);
    ColVectorXi64 Y = Yi.segment(indexLo, BATCH_SIZE);
    RowMatrixXi64 Eb = E.block(indexLo, 0, BATCH_SIZE, d);
    ColVectorXi64 V = Vi.col(iter);
    ColVectorXi64 V_ = Vi_.col(iter);
    ColVectorXi64 Z = Zi.col(iter);
    ColVectorXi64 Z_ = Zi_.col(iter);

    std::cout << "[train_batch] X shape: " << X.rows() << "x" << X.cols() << ", Y size: " << Y.size() << std::endl;
    std::cout << "[train_batch] Eb shape: " << Eb.rows() << "x" << Eb.cols() << std::endl;

    Fi = wi - V;
    std::cout << "[train_batch] Fi computed." << std::endl;

    if(party == ALICE) {
        send<ColVectorXi64>(io, Fi);
        std::cout << "[train_batch] ALICE sent Fi." << std::endl;
    }else {
        recv<ColVectorXi64>(io, F);
        std::cout << "[train_batch] BOB received Fi." << std::endl;
    }

    if(party == BOB) {
        send<ColVectorXi64>(io, Fi);
        std::cout << "[train_batch] BOB sent Fi." << std::endl;
    }else {
        recv<ColVectorXi64>(io, F);
        std::cout << "[train_batch] ALICE received Fi." << std::endl;
    }

    F += Fi;
    std::cout << "[train_batch] F updated." << std::endl;

    ColVectorXi64 Y_(BATCH_SIZE);
    ColVectorXi64 D(BATCH_SIZE);
    ColVectorXi64 Fi_(BATCH_SIZE);
    ColVectorXi64 F_(BATCH_SIZE);
    ColVectorXi64 delta(d);

    Y_ = -iter * (Eb * F) + X * F + Eb * wi + Zi;
    std::cout << "[train_batch] Y_ computed." << std::endl;

    truncate<ColVectorXi64>(iter, SCALING_FACTOR, Y_);
    std::cout << "[train_batch] Y_ truncated." << std::endl;

    D = Y_ - Y;
    Fi_ = D - V_;
    std::cout << "[train_batch] D and Fi_ computed." << std::endl;

    if (party == ALICE) {
        send<ColVectorXi64>(io, Fi_);
        std::cout << "[train_batch] ALICE sent Fi_." << std::endl;
    } else {
        recv<ColVectorXi64>(io, F_);
        std::cout << "[train_batch] BOB received Fi_." << std::endl;
    }

    if (party == BOB) {
        send<ColVectorXi64>(io, Fi_);
        std::cout << "[train_batch] BOB sent Fi_." << std::endl;
    } else {
        recv<ColVectorXi64>(io, F_);
        std::cout << "[train_batch] ALICE received Fi_." << std::endl;
    }

    F_ += Fi_;
    std::cout << "[train_batch] F_ updated." << std::endl;

    RowMatrixXi64 Ebt = Eb.transpose();
    RowMatrixXi64 Xt = X.transpose();

    delta = -iter * (Ebt * F_) + Xt * F_ + Ebt * D + Z_;
    std::cout << "[train_batch] delta computed." << std::endl;

    truncate<ColVectorXi64>(iter, SCALING_FACTOR, delta);
    truncate<ColVectorXi64>(iter, alpha_inv * BATCH_SIZE, delta);
    std::cout << "[train_batch] delta truncated." << std::endl;

    wi -= delta;
    std::cout << "[train_batch] wi updated." << std::endl;
}
