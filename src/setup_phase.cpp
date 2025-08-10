#include "setup_phase.hpp"

using namespace Eigen;
using namespace emp;
using namespace std;

void SetupPhase::initialize() {
    prg.random_data(Ai.data(), n * d * 8);
    prg.random_data(Bi.data(), d * t * 8);
    prg.random_data(Bi_.data(), BATCH_SIZE * t * 8);
    std::cout << "Matrices Initialized" << std::endl;
}

void SetupPhase::generateMTs() {
    cout << "Party " << party << ": Starting generateMTs()" << endl;
    
    vector<vector<uint64_t>> ci(t, vector<uint64_t>(BATCH_SIZE));
    vector<vector<uint64_t>> ci_(t, vector<uint64_t>(d));
    for(int i = 0; i < t; i++) {
        cout << "Party " << party << ": Processing batch " << i << "/" << t << endl;
        
        RowMatrixXi64 Ai_b = Ai.block(i * BATCH_SIZE, 0, BATCH_SIZE, d);
        vector<vector<uint64_t>> ai(BATCH_SIZE, vector<uint64_t>(d));
        RowMatrixXi64_to_vector2d(Ai_b, ai);

        RowMatrixXi64 Ai_bt = Ai_b.transpose();
        vector<vector<uint64_t>> ai_t(d, vector<uint64_t>(BATCH_SIZE));
        RowMatrixXi64_to_vector2d(Ai_bt, ai_t);

        vector<uint64_t> bi = ColVectorXi64_to_vector(Bi.col(i));
        vector<uint64_t> bi_ = ColVectorXi64_to_vector(Bi_.col(i));

        secure_mult(BATCH_SIZE, d, ai, bi, ci[i]);

        secure_mult(d, BATCH_SIZE, ai_t, bi_, ci_[i]);
    }  
    
    vector2d_to_ColMatrixXi64(ci, Ci);
    vector2d_to_ColMatrixXi64(ci_, Ci_); 
    cout << "Party " << party << ": Triples Generated" << endl;

#if DEBUG
    verify();
#endif
}

void SetupPhase::secure_mult(int n, int d, vector<vector<uint64_t>>& a, vector<uint64_t>& b, vector<uint64_t>& c){
    cout << "Party " << party << ": secure_mult started with n=" << n << ", d=" << d << endl;
    
    // NUM_OT[p]: the number of OT at p-th bit position
    int NUM_OT[BITLEN]; 

    // Calculate the total number of OT
    int total_ot = 0;
    for(int p = 0; p < BITLEN; p++) {
        int temp = 128/(64 - p); 
        NUM_OT[p] = n/temp;
        if (n % temp) {
            NUM_OT[p]++;
        }
        total_ot += NUM_OT[p];
    }
    total_ot *= d;

    // m0[y]: y-th OT message m0 of sender
    // m1[y]: y-th OT message m1 of sender
    // rec[y]: y-th received message of receiver
    block *m0, *m1, *rec;
    m0 = new block[total_ot]; 
    m1 = new block[total_ot]; 
    rec = new block[total_ot]; 
    int index_m0 = 0;
    int index_m1 = 0;

    // bits_b[p]: the p-th bit of b[j]
    // sigma[y]: the y-th choosing bit of the receiver
    bool bits_b[BITLEN];
    bool* sigma;
    sigma = new bool[total_ot];
    int index_sigma = 0;

    // Calculate the random mask for each bit position
    uint64_t*** X0;
    X0 = new uint64_t**[n];
    for(int i = 0; i < n; i++) {
        X0[i] = new uint64_t* [d];
        for(int j = 0; j < d; j++) {
            X0[i][j] = new uint64_t [BITLEN];
            prg.random_data(X0[i][j], BITLEN * 8);
        }
    }
    
    for(int j = 0; j < d; j++) {
        int_to_bool(b[j], bits_b);

        for(int p = 0; p < BITLEN; p++) {
            uint64_t X1[n]; 
            for(int i = 0; i < n; i++) {
                X1[i] = X0[i][j][p] + a[i][j];
            }
            int index_row = 0;
            int elements_in_block = 128/(64-p);

            for(int y = 0; y < NUM_OT[p]; y++) {
                sigma[index_sigma++] = bits_b[p];
            }

            for(int y = 0; y < NUM_OT[p]; y++) {
                int flag = elements_in_block;
                uint64_t temp_lo = 0, temp_hi = 0;
                uint64_t r_temp_lo = 0, r_temp_hi = 0;
                int elements_in_temp = 64/(64-p);
                int left_bitsLo = (64 % ((64-p)*elements_in_temp));

                temp_lo = (X1[index_row] << p);
                temp_lo >>= p;
                r_temp_lo = (X0[index_row++][j][p] << p);
                r_temp_lo >>= p;
                flag--;
                for(int z = 1; z < elements_in_temp; z++) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t r_next_element = (X0[index_row][j][p] << p);
                        r_next_element >>= p;
                        r_next_element <<= ((64-p) * z);
                        r_temp_lo ^= r_next_element;
                        uint64_t next_element = (X1[index_row++] << p);
                        next_element >>= p;
                        next_element <<= ((64-p) * z);
                        temp_lo ^= next_element;
                        flag--;
                    }else {
                        break;
                    }
                }

                if(left_bitsLo) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t r_split_element = (X0[index_row][j][p] << p);
                        r_split_element >>= p;
                        r_temp_lo ^= (r_split_element << (64-left_bitsLo));
                        r_temp_hi ^= (r_split_element >> left_bitsLo);
                        uint64_t split_element = (X1[index_row++] << p);
                        split_element >>= p;
                        temp_lo ^= (split_element << (64-left_bitsLo));
                        temp_hi ^= (split_element >> left_bitsLo);
                        flag--;
                    }else {
                        break;
                    }
                }

                for(int z = 0; z < elements_in_temp; z++) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t r_next_element = (X0[index_row][j][p] << p);
                        r_next_element >>= p;
                        if(left_bitsLo) {
                            r_next_element <<= ((64-p)*z+(64-p-left_bitsLo)); 
                        }else {
                            r_next_element <<= ((64-p)*z);
                        }
                        r_temp_hi ^= r_next_element;
                        uint64_t next_element = (X1[index_row++] << p);
                        next_element >>= p;
                        if(left_bitsLo) {
                            next_element <<= ((64-p)*z+(64-p-left_bitsLo));
                        }else {
                            next_element <<= ((64-p)*z);
                        }
                        temp_hi ^= next_element;
                        flag--;
                    }else {
                        break;
                    }
                }
                m0[index_m0++] = makeBlock(r_temp_lo, r_temp_hi);
                m1[index_m1++] = makeBlock(temp_lo, temp_hi);
            }

        }
    }

    cout << "Party " << party << ": Starting OT operations with total_ot=" << total_ot << endl;
    
    if(party == ALICE) {
        cout << "Party 1: Sending first OT..." << endl;
        send_ot->send(m0, m1, total_ot);
    }else {
        cout << "Party " << party << ": Receiving first OT..." << endl;
        recv_ot->recv(rec, sigma, total_ot);
    }

    if(party == BOB) {
        cout << "Party 2: Sending second OT..." << endl;
        send_ot->send(m0, m1, total_ot);
    }else {
        cout << "Party " << party << ": Receiving second OT..." << endl;
        recv_ot->recv(rec, sigma, total_ot);
    }
    
    cout << "Party " << party << ": OT operations completed" << endl;

 

    int indexRec = 0;
    for(int j = 0; j < d; j++) {
        
        for(int p = 0; p < BITLEN; p++) {
            int index_row = 0;
            int elements_in_block = 128/(64-p);

            for(int y = 0; y < NUM_OT[p]; y++) {
                int flag = elements_in_block;
                uint64_t temp_lo =  extract_lo64(rec[indexRec]);
                uint64_t temp_hi =  extract_hi64(rec[indexRec++]);

                int elements_in_temp = 64/(64-p);
                int left_bitsLo = (64 % ((64-p) * elements_in_temp));
                uint64_t mask;
                if(p > 0) {
                    mask = ((1ULL << (64 - p)) - 1);
                }else {
                    mask = -1;
                }

                for(int z = 0; z < elements_in_temp; z++) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t next_element = (temp_lo & mask);
                        next_element <<= p;
                        c[index_row++] += next_element;
                        temp_lo >>= (64 - p);
                        flag--;
                    } else {
                        break;
                    }
                }
                if(left_bitsLo) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t split_mask;
                        split_mask = ((1ULL << (64-p-left_bitsLo)) -1);
                        uint64_t next_element = temp_hi & split_mask;
                        next_element <<= left_bitsLo;
                        next_element ^= temp_lo;
                        next_element <<= p;
                        c[index_row++] += next_element;
                        temp_hi >>= (64 - p - left_bitsLo);
                        flag--;
                    } else {
                        break;
                    }
                }
                for(int z = 0; z < elements_in_temp; z++) {
                    if((index_row <= n - 1) && flag) {
                        uint64_t next_element = (temp_hi & mask);
                        next_element <<= p;
                        c[index_row++] += next_element;
                        temp_hi >>= (64 - p);
                        flag--;
                    } else {
                        break;
                    }
                }
            }

            for(int i = 0; i < n; i++) {
                c[i] -= (X0[i][j][p] << p);
            }
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < d; j++) {
            delete X0[i][j];
        }
        delete X0[i];
    }
    delete X0;
}

void SetupPhase::verify() {

    RowMatrixXi64 Ai_b(BATCH_SIZE, d);
    ColMatrixXi64 Bi_b(d, 1);
    ColMatrixXi64 Ci_b(BATCH_SIZE, 1);
    ColMatrixXi64 Bi_b_(BATCH_SIZE, 1);
    ColMatrixXi64 Ci_b_(d, 1);

    if(party == 1) {
        for(int i = 0; i < t; i++) {
            Ai_b = Ai.block(i*BATCH_SIZE, 0, BATCH_SIZE, d);
            Bi_b = Bi.col(i);
            Ci_b = Ci.col(i);
            Bi_b_ = Bi_.col(i);
            Ci_b_ = Ci_.col(i);

            send(io, Ai_b);
            send(io, Bi_b);
            send(io, Ci_b);
            send(io, Bi_b_);
            send(io, Ci_b_);
        }
    } else {
        bool flag = true;
        bool flag_ = true;
        RowMatrixXi64 A(BATCH_SIZE, d);
        ColMatrixXi64 B(d, 1);
        ColMatrixXi64 AB(BATCH_SIZE, 1);
        ColMatrixXi64 C(BATCH_SIZE, 1);
        RowMatrixXi64 A_t(d, BATCH_SIZE);
        ColMatrixXi64 B_(BATCH_SIZE, 1);
        ColMatrixXi64 AB_(d, 1);
        ColMatrixXi64 C_(d, 1);
        for(int i = 0; i < t; i++) {
            Ai_b = Ai.block(i*BATCH_SIZE, 0, BATCH_SIZE, d);
            Bi_b = Bi.col(i);
            Ci_b = Ci.col(i);
            Bi_b_ = Bi_.col(i);
            Ci_b_ = Ci_.col(i);

            recv(io, A);
            recv(io, B);
            recv(io, C);
            recv(io, B_);
            recv(io, C_);

            A += Ai_b;
            A_t = A.transpose();
            B += Bi_b;
            C += Ci_b;
            B_ += Bi_b_;
            C_ += Ci_b_;

            AB = A * B;
            AB_ = A_t * B_;

        }

        if(flag == true && flag_ == true) {
            cout << "Verification Successful" << endl;
        }else {
            cout << "Verification failed" << endl;
        }
    }
}

void SetupPhase::getMTs(SetupTriples *triples) {
    triples->Ai = this->Ai;
    triples->Bi = this->Bi;
    triples->Ci = this->Ci;
    triples->Bi_ = this-> Bi_;
    triples->Ci_ = this-> Ci_;
}