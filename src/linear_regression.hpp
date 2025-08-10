#ifndef SECML_LR_HPP
#define SECML_LR_HPP

#include "setup_phase.hpp"
#include "online_phase.hpp"

class LinearRegression {
public:
    emp::NetIO* io;
    int party;
    int n, d, t;
    RowMatrixXi64 X;
    ColVectorXi64 Y;
    ColVectorXi64 w; // the weights during training 
    ColVectorXd w_d; // the weights after training 
    SetupPhase* setup;
    OnlinePhase* online;
    LinearRegression(RowMatrixXi64& training_data, ColVectorXi64& training_labels, 
                    TrainingParams params, emp::NetIO* io) {
        std::cout << "[LinearRegression] Constructing, party=" << party << ", n=" << params.n << ", d=" << params.d << std::endl;
        this->n = params.n;
        this->d = params.d;
        this->t = (params.n)/BATCH_SIZE; // number of iterations
        this->X = training_data;
        this->Y = training_labels;
        this->io = io;
        this->party = PARTY;
        this->w.resize(d);  
        this->w_d.resize(d); 

        std::cout << "[LinearRegression] Creating SetupPhase..." << std::endl;
        this->setup = new SetupPhase(n, d, t, io);
        
        std::cout << "[LinearRegression] Calling setup->initialize()..." << std::endl;
        setup->initialize();
        std::cout << "[LinearRegression] Calling setup->generateMTs()..." << std::endl;
        setup->generateMTs();
        std::cout << "[LinearRegression] Setup done" << std::endl;
        SetupTriples triples;
        std::cout << "[LinearRegression] Calling setup->getMTs()..." << std::endl;
        setup->getMTs(&triples);

        RowMatrixXi64 Xi(X.rows(), X.cols()); // the secret sharing of X
        ColVectorXi64 Yi(Y.rows(), Y.cols()); // the secret sharing of Y
        if(party == emp::ALICE) {
            emp::PRG prg;
            RowMatrixXi64 rX(X.rows(), X.cols());
            ColVectorXi64 rY(Y.rows(), Y.cols());
            std::cout << "[LinearRegression] ALICE generating random mask..." << std::endl;
            prg.random_data(rX.data(), X.rows() * X.cols() * sizeof(uint64_t));
            prg.random_data(rY.data(), Y.rows() * Y.cols() * sizeof(uint64_t));
            
            Xi = X + rX;
            Yi = Y + rY;
            rX *= -1;
            rY *= -1;
            std::cout << "[LinearRegression] ALICE sending rX and rY..." << std::endl;
            send<RowMatrixXi64>(io, rX);
            send<ColVectorXi64>(io, rY);
            std::cout << "[LinearRegression] ALICE sent rX and rY." << std::endl;
        }else {
            std::cout << "[LinearRegression] BOB waiting to receive rX and rY..." << std::endl;
            recv<RowMatrixXi64>(io, Xi);
            recv<ColVectorXi64>(io, Yi);
            std::cout << "[LinearRegression] BOB received rX and rY." << std::endl;
        }
        std::cout << "[LinearRegression] Creating OnlinePhase..." << std::endl;
        this-> online = new OnlinePhase(params, io, &triples);
        std::cout << "[LinearRegression] Initializing OnlinePhase..." << std::endl;
        online->initialize(Xi, Yi);    

        std::cout << "[LinearRegression] Calling train_model..." << std::endl;
        train_model();
    }
    void initialize();
    void train_model();
    void test_model(RowMatrixXd& testing_data, ColVectorXd& testing_labels);  
        
};



#endif