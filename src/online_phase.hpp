#ifndef SECML_ONLINE_HPP
#define SECML_ONLINE_HPP
#include <math.h>
#include "util.hpp"

class OnlinePhase{
public:
    int party;
    int n, d, t, i, alpha_inv;
    SetupTriples* triples;
    emp::PRG prg;
    emp::NetIO* io;

    RowMatrixXi64 Xi, Ui, Ei, E;
    ColMatrixXi64 Vi, Zi, Vi_, Zi_;
    ColVectorXi64 Yi, wi, Fi, F;

    OnlinePhase(TrainingParams params, emp::NetIO* io, SetupTriples* triples) {
        this->n = params.n;
        this->d = params.d;
        this->t = (params.n)/BATCH_SIZE;
        this->party = PARTY;
        this->alpha_inv = params.alpha_inv;
        this->io = io;
        this->triples = triples;
        
        if(party == emp::ALICE) {
            i = 0;
        }else {
            i = 1;
        }

        Xi.resize(n, d);
        Ui.resize(n, d);
        Ei.resize(n, d);
        E.resize(n, d);
        
        Vi.resize(d, t);
        Zi.resize(BATCH_SIZE, t);
        Vi_.resize(BATCH_SIZE, t);
        Zi_.resize(d, t);

        Yi.resize(n);
        wi.resize(d);
        Fi.resize(d);
        F.resize(d);
    }

    void initialize(RowMatrixXi64& Xi, ColVectorXi64& Yi);
    void train_batch(int iter, int indexLo);
};

#endif