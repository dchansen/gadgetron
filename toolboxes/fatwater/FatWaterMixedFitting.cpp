//
// Created by dch on 18/04/18.
//

#include "FatWaterMixedFitting.h"
#include "fatwater.h"

#include <numeric>
#include <boost/math/constants/constants.hpp>
#include <limits>
#include "complext.h"


namespace std {
    template<> class numeric_limits<std::complex<float>> : public numeric_limits<float>{


    };
}



using namespace Gadgetron;
static constexpr float GAMMABAR = 42.576;
using namespace std::complex_literals;
constexpr double PI = boost::math::constants::pi<double>();

#include <ceres/ceres.h>




struct complex_residual {
    template< class T> void operator()(T* e, const complext<T>& predicted, const complext<double>& data, int index) const {
        auto residual = predicted-data;
        e[2*index] = residual.real();
        e[2*index+1] = residual.imag();
    }
};


struct abs_residual {
    template< class T> void operator()(T* e, const complext<T>& predicted, const complext<double>& data, int index) const {
        auto residual = abs(predicted)-abs(data);
        e[index] = residual;
    }
};

template<unsigned int NSPECIES,unsigned int NRESIDUALS, class RESIDUAL> class FatWaterModelCeres {
public:


    FatWaterModelCeres(FatWaterAlgorithm alg, const std::vector<float>& TEs, const std::vector<complext<double>>& data, float fieldstrength,float r2star) : alg_(alg), TEs_(TEs),data_(data), fieldstrength_(fieldstrength), residual_function(), r2star_(r2star){
        assert(alg_.species_.size() == NSPECIES);
        assert(TEs.size() == data_.size());
        assert(data_.size() == NRESIDUALS);


        auto calc_amp = [&](auto& species, auto TE){
            return std::accumulate(species.ampFreq_.begin(), species.ampFreq_.end(), complext<double>(0.0),
                            [&](auto val, auto peak) {
                                return val + complext<float>(peak.first) * exp(complext<double>(0.0,2.0) * (PI * peak.second*fieldstrength_*GAMMABAR * TE));
                            });
        };




        auto fat = alg.species_[1];
        auto water = alg.species_[0];
        for (int j = 0; j < NRESIDUALS; j++){
            const float TE = TEs[j];
            fat_amp[j] = calc_amp(fat,TE);
            water_amp[j] = calc_amp(water,TE);
        }

    };

    template<class T>
    bool operator()(const T* const fm_ptr, const T* const water_ptr, const T* const fat_ptr, T* e) const {


        const T& fm = *fm_ptr;
//        const T& r2star = *r2star_ptr;
//        T fm = T(0.0);
//        T r2star = T(0.0);
        const double r2star = r2star;
//        float fm = 0;
//        T r2star = T(0);
        auto water = complext<T>(water_ptr[0],water_ptr[1]);
        auto fat = complext<T>(fat_ptr[0],fat_ptr[1]);

        for (int j = 0; j < NRESIDUALS; j++) {
            const float TE = TEs_[j];
            complext<T> predicted = fat*fat_amp[j]+water*water_amp[j];

            predicted *= exp((T(-r2star) + complext<T>(T(0),T(2*PI)) * fm) * T(TE));
            residual_function(e, predicted, data_[j],j);

        }
        return true;


    }




private:
    const FatWaterAlgorithm alg_;

    const std::vector<float> TEs_;
    const std::vector<complext<double>> data_;
    const float fieldstrength_;
    const float r2star_;
    RESIDUAL residual_function;
    std::array<complext<double>,NRESIDUALS> fat_amp;
    std::array<complext<double>,NRESIDUALS> water_amp;


};




//void Gadgetron::fat_water_mixed_fitting(hoNDArray<float> &field_map, hoNDArray<float> &r2star_map,
//                                        hoNDArray<std::complex<float>> &fractions,
//                                        const hoNDArray<std::complex<float>> &input_data,
//                                        const FatWaterAlgorithm &alg_, const std::vector<float> &TEs, float fieldstrength){
//
//
//
//
//    const size_t X = input_data.get_size(0);
//    const size_t Y = input_data.get_size(1);
//    const size_t N = input_data.get_size(4);
//    const size_t S = input_data.get_size(5);
//
//    std::vector<float> TEs_repeated((S)*N);
//
//
//
//
//
//
//    for (int k3 = 0; k3 < S; k3++) {
//        for (int k4 = 0; k4 < N; k4++) {
//            TEs_repeated[k4 + (k3) * N] = float(TEs[k3]);
//        }
//    }
//
//
//#pragma omp parallel for collapse(2)
//    for (int k2 = 0; k2 < Y; k2++){
//        for (int k1 = 0; k1 < X; k1++){
//
//            std::vector<complext<double>> signal((S-1)*N);
//            auto& f = field_map(k1,k2);
//            auto& r2 = r2star_map(k1,k2);
//            std::complex<float>& water = fractions(k1,k2,0,0,0,0,0);
//            std::complex<float>& fat =   fractions(k1,k2,0,0,0,1,0);
//
//            for (int k3 = 1; k3 < S; k3++) {
//                for (int k4 = 0; k4 < N; k4++) {
//                    signal[k4+(k3)*N] = input_data(k1, k2, 0, 0, k4, k3, 0);
//                }
//            }
//
//            ceres::Problem problem;
//            /*
//            auto cost_function1 = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,1,abs_residual>,ceres::RIDDERS,1,6>(
//                    new FatWaterModelCeres<2,1,abs_residual>(alg_,TEs_repeated1,signal1,fieldstrength));
//
//
//
//            auto cost_function = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,3,complex_residual>,ceres::RIDDERS,6,6>(
//                    new FatWaterModelCeres<2,3,complex_residual>(alg_,TEs_repeated,signal,fieldstrength));
//                    */
//
//
//
//
//
//
//            auto cost_function = new ceres::AutoDiffCostFunction<FatWaterModelCeres<2,4,complex_residual>,8,6>(
//                    new FatWaterModelCeres<2,3,complex_residual>(alg_,TEs_repeated,signal,fieldstrength,r2));
//
////            std::vector<double> b = {f,r2,water.real(),water.imag(),fat.real(),fat.imag()};
//
//                        problem.AddResidualBlock(cost_function, nullptr, b.data());
////            problem.SetParameterLowerBound(b.data(),1,0);
////            problem.SetParameterUpperBound(b.data(),1,200);
//
//            ceres::Solver::Options options;
//           options.max_num_iterations = 50;
//            options.linear_solver_type = ceres::DENSE_QR;
////            options.initial_trust_region_radius = 0.1;
////            options.dense_linear_algebra_library_type = ceres::LAPACK;
////            options.use_inner_iterations = true;
//
////        options.use_explicit_schur_complement = true;
////            options.function_tolerance = 1e-4;
////            options.gradient_tolerance = 1e-4;
////            options.parameter_tolerance = 1e-4;
////        options.preconditioner_type = ceres::IDENTITY;
////    options.minimizer_type = ceres::LINE_SEARCH;
////         options.line_search_direction_type = ceres::LBFGS;
////        options.trust_region_strategy_type = ceres::DOGLEG;
////    options.dogleg_type = ceres::SUBSPACE_DOGLEG;
//
//
//            ceres::Solver::Summary summary;
//            ceres::Solve(options, &problem, &summary);
//
//
//
//            f = b[0];
//            r2 = b[1];
//            water = std::complex<float>(b[2],b[3]);
//            fat = std::complex<float>(b[4],b[5]);
//
//        }
//    }
//
//
//
//}
//
//
//

struct DiffLoss {
    DiffLoss(double scale1): scale1_(scale1){}
    template<class T> bool operator()(const T* const base, const T* const dx,  T* residual) const{

        residual[0] = scale1_*(base[0]-dx[0]);

        return true;
    }

private:
    const double scale1_;

};


static void add_regularization(ceres::Problem & problem, hoNDArray<double>& field_map, const hoNDArray<float>& lambda_map, ceres::LossFunction* loss=NULL){

    const size_t X = field_map.get_size(0);
    const size_t Y = field_map.get_size(1);

    for (int k2 = 0; k2 < Y-1; k2++){
        for (int k1 = 0; k1 < X-1; k1++){

            auto weight1 = std::min(lambda_map(k1,k2),lambda_map(k1+1,k2));
            auto weight2 = std::min(lambda_map(k1,k2),lambda_map(k1,k2+1));
            {
                auto cost_function = new ceres::AutoDiffCostFunction<DiffLoss,1, 1, 1>(new DiffLoss(weight1));
                std::vector<double *> ptrs = {&field_map(k1, k2), &field_map(k1 + 1, k2)};

                problem.AddResidualBlock(cost_function, loss, ptrs);
            }
            {
                auto cost_function = new ceres::AutoDiffCostFunction<DiffLoss, 1, 1, 1>(new DiffLoss(weight2));
                std::vector<double *> ptrs = {&field_map(k1, k2), &field_map(k1, k2+1)};

                problem.AddResidualBlock(cost_function, loss, ptrs);
            }
        }
    }
}

/*
static void add_regularization(ceres::Problem & problem, hoNDArray<double>& field_map, float lambda, ceres::LossFunction* loss=NULL){

    const size_t X = field_map.get_size(0);
    const size_t Y = field_map.get_size(1);

    for (int k2 = 0; k2 < Y-1; k2++){
        for (int k1 = 0; k1 < X-1; k1++){


            auto cost_function = new ceres::AutoDiffCostFunction<DiffLoss,2,1,1,1>(new DiffLoss(lambda,lambda));
            std::vector<double*> ptrs = {&field_map(k1,k2),&field_map(k1+1,k2),&field_map(k1,k2+1)};

            problem.AddResidualBlock(cost_function,loss,ptrs);
        }
    }
}*/


void Gadgetron::fat_water_mixed_fitting(hoNDArray<float> &field_mapF, hoNDArray<float> &r2star_mapF,
                                        hoNDArray<std::complex<float>> &fractionsF,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const hoNDArray<float> &lambda_map, const FatWaterAlgorithm &alg_,
                                        const std::vector<float> &TEs, float fieldstrength) {

    hoNDArray<double> field_map;
    field_map.copyFrom(field_mapF);
    hoNDArray<double > r2star_map;
    r2star_map.copyFrom(r2star_mapF);
    hoNDArray<std::complex<double>> fractions;
    fractions.copyFrom(fractionsF);


    const size_t X = input_data.get_size(0);
    const size_t Y = input_data.get_size(1);
    const size_t N = input_data.get_size(4);
    const size_t S = input_data.get_size(5);

    std::vector<float> TEs_repeated((S)*N);






    for (int k3 = 0; k3 < S; k3++) {
        for (int k4 = 0; k4 < N; k4++) {
            TEs_repeated[k4 + (k3) * N] = double(TEs[k3]);
        }
    }

    ceres::Problem problem;
    ceres::Solver::Options options;
//    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 6;
//    options.minimizer_type = ceres::LINE_SEARCH;
//    options.line_search_direction_type = ceres::LBFGS;
//    options.nonlinear_conjugate_gradient_type = ceres::POLAK_RIBIERE;
//    options.max_lbfgs_rank = 10;
//    options.use_nonmonotonic_steps = true;
//    options.use_inner_iterations = true;
//    options.inner_iteration_tolerance = 1e-2;
//    options.inner_iteration_tolerance = 1e-5;

//     options.trust_region_strategy_type = ceres::DOGLEG;
//     options.dogleg_type = ceres::SUBSPACE_DOGLEG;
    for (int k2 = 0; k2 < Y; k2++){
        for (int k1 = 0; k1 < X; k1++){

            std::vector<complext<double>> signal((S)*N);
            auto& f = field_map(k1,k2);
            auto& r2 = r2star_map(k1,k2);
            std::complex<double>& water = fractions(k1,k2,0,0,0,0,0);
            std::complex<double>& fat =   fractions(k1,k2,0,0,0,1,0);

            for (int k3 = 1; k3 < S; k3++) {
                for (int k4 = 0; k4 < N; k4++) {
                    signal[k4+(k3)*N] = input_data(k1, k2, 0, 0, k4, k3, 0);
                }
            }

            /*
            auto cost_function1 = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,1,abs_residual>,ceres::RIDDERS,1,6>(
                    new FatWaterModelCeres<2,1,abs_residual>(alg_,TEs_repeated1,signal1,fieldstrength));



            auto cost_function = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,3,complex_residual>,ceres::RIDDERS,6,6>(
                    new FatWaterModelCeres<2,3,complex_residual>(alg_,TEs_repeated,signal,fieldstrength));
                    */






            auto cost_function = new ceres::AutoDiffCostFunction<FatWaterModelCeres<2,4,complex_residual>,8,1,2,2>(
                    new FatWaterModelCeres<2,4,complex_residual>(alg_,TEs_repeated,signal,fieldstrength,r2));

//            std::vector<double> b = {f,r2,water.real(),water.imag(),fat.real(),fat.imag()};
            std::vector<double*> b = {&f, (double*)&water,(double*)&fat};
            problem.AddResidualBlock(cost_function, nullptr, b);
//            problem.SetParameterLowerBound(&r2,0,0);
//            problem.SetParameterUpperBound(&r2,0,500);
//

//            options.initial_trust_region_radius = 0.1;
//            options.dense_linear_algebra_library_type = ceres::LAPACK;
//            options.use_inner_iterations = true;

//        options.use_explicit_schur_complement = true;
//            options.function_tolerance = 1e-4;
//            options.gradient_tolerance = 1e-4;
//            options.parameter_tolerance = 1e-4;
//        options.preconditioner_type = ceres::IDENTITY;
//    options.minimizer_type = ceres::LINE_SEARCH;
//         options.line_search_direction_type = ceres::LBFGS;
//        options.trust_region_strategy_type = ceres::DOGLEG;
//    options.dogleg_type = ceres::SUBSPACE_DOGLEG;








        }
    }
//    add_regularization(problem,field_map,lambda_map);
//    add_regularization(problem,field_map,2);
//    hoNDArray<
//    add_regularization(problem,r2star_map,1.0, new ceres::SoftLOneLoss(2));

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    field_mapF.copyFrom(field_map);
    r2star_mapF.copyFrom(r2star_map);
    fractionsF.copyFrom(fractions);


}




