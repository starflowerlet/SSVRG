#include "grad_desc_async_sparse.hpp"
#include "utils.hpp"
#include "regularizer.hpp"
#include <atomic>
#include <random>
#include <cmath>
#include <thread>
#include <mutex>
#include <string.h>
#include<chrono>
using namespace std;
extern size_t MAX_DIM;
std::atomic<int> S2VRG_counter(0);
std::atomic<int> S2VRG_counter_1(1);

chrono::steady_clock::time_point start_S2VRG = chrono::steady_clock::now();

void grad_desc_async_sparse::S2VRG_Inner_Loop(double* X, double* Y, size_t* Jc
    , size_t* Ir, size_t N, std::atomic<double>* x, std::atomic<double>* aver_x
    , blackbox* model, size_t m, size_t inner_iters, double step_size
    , std::atomic<double>* reweight_diag, double* full_grad_core, double* full_grad
    , double* x_tilda,std::vector<double>* losses,  size_t thread_no,  size_t _thread) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> distribution(0, N - 1);
    std::uniform_int_distribution<int> R_distribution(0,thread_no-1);
    int regular = model->get_regularizer();
    int iter_no,iter_no_1;
    double* lambda = model->get_params();
    double* inconsis_x = new double[MAX_DIM];
    for(size_t j = 0; j < inner_iters; j ++) {
        iter_no = S2VRG_counter.fetch_add(1);
        int rand_samp = distribution(generator);
        // Inconsistant Read [X].
        for(size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k ++)
            inconsis_x[Ir[k]] = x[Ir[k]];
        double inner_core = model->first_component_oracle_core_sparse(X, Y
                , Jc, Ir, N, rand_samp, inconsis_x);
        int R_random = R_distribution(generator);
		for (size_t k = Jc[rand_samp]; k < Jc[rand_samp + 1]; k++) {
			size_t index = Ir[k];
 			if (index > MAX_DIM / thread_no*(_thread-1) && index <= MAX_DIM / thread_no * _thread
                    || index >  MAX_DIM / thread_no*(R_random-1) && index <= MAX_DIM / thread_no * R_random)
			{
				double val = X[k];
				double vr_sub_grad = ((inner_core - full_grad_core[rand_samp]) * val
					+ reweight_diag[index] 
                    * (full_grad[index]+ lambda[0] * inconsis_x[index]));
				double incr_x = -step_size * vr_sub_grad;
				// Atomic Write
// 				fetch_n_add_atomic(x[index], incr_x);
// 				fetch_n_add_atomic(aver_x[index], incr_x * (m + 1 - iter_no) / m);

				//NON-Atomic 
				x[index] = x[index] + incr_x;
				aver_x[index] = aver_x[index] + incr_x * (m + 1 - iter_no) / m;
			}
        }
        
    }
    delete[] inconsis_x;
}

std::vector<double>* grad_desc_async_sparse::S2VRG_Async(double* X
    , double* Y, size_t* Jc, size_t* Ir, size_t N, blackbox* model
    , size_t iteration_no, size_t thread_no, double L
    , double step_size, bool is_store_result) {
    S2VRG_counter_1=1;
    std::vector<double>* losses = new std::vector<double>;
    std::atomic<double>* x = new std::atomic<double>[MAX_DIM];
    // "Anticipate" Update Extra parameters
    std::atomic<double>* reweight_diag = new std::atomic<double>[MAX_DIM];
    // Average Iterates
    std::atomic<double>* aver_x = new std::atomic<double>[MAX_DIM];
    size_t m = N/4;
    memset(reweight_diag, 0, MAX_DIM * sizeof(double));
    copy_vec((double *)x, model->get_model());
    // Init Weight Evaluate
    if(is_store_result) {
		chrono::steady_clock::time_point end = chrono::steady_clock::now();
        losses->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
		chrono::duration<double>time_used = chrono::duration_cast<chrono::duration<double>>(end - start_S2VRG);
		losses->push_back(time_used.count());
    }

    // OUTTER_LOOP
    for(size_t i = 0 ; i < iteration_no; i ++) {
        double* outter_x = model->get_model();
        double* full_grad_core = new double[N];
        double* full_grad;
		S2VRG_counter = 0;
        
        // Full Gradient
        if(i == 0) {
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x, reweight_diag);
            // Compute Re-weight Matrix in First Pass
            for(size_t j = 0; j < MAX_DIM; j ++)
                reweight_diag[j] = 1.0 / reweight_diag[j];
        }
        else
            full_grad = Comp_Full_Grad_Parallel(full_grad_core, thread_no
                , X, Y, Jc, Ir, N, model, outter_x);

        copy_vec((double *)aver_x, (double *)x);
        // Parallel INNER_LOOP
        std::vector<std::thread> thread_pool;
        size_t imax=3;
        size_t m_i;
        if(i<imax)
            m_i = pow(2,i)* m ;
         else
            m_i = pow(2,imax)*m;
        for(size_t k = 1; k <= thread_no; k ++) {
            size_t inner_iters;
            inner_iters = (double)m_i / thread_no;
            thread_pool.push_back(std::thread(S2VRG_Inner_Loop, X, Y, Jc, Ir, N
                , x, aver_x, model, m_i, inner_iters, step_size, reweight_diag
                , full_grad_core, full_grad, outter_x,losses, thread_no, k));
        }
        for(auto& t : thread_pool)
            t.join();

        model->update_model((double*) aver_x);

        // For Matlab (per m/n passes)
         if(is_store_result) {
			 chrono::steady_clock::time_point end = chrono::steady_clock::now();
			 losses->push_back(model->zero_oracle_sparse(X, Y, Jc, Ir, N));
			 chrono::duration<double>time_used = chrono::duration_cast<chrono::duration<double>>(end - start_S2VRG);
			 losses->push_back(time_used.count());
         }
        delete[] full_grad_core;
        delete[] full_grad;
    }
    delete[] reweight_diag;
    delete[] x;
    delete[] aver_x;
    return losses;
}
