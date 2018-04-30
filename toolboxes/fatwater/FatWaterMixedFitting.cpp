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
    };

    template<class T>
    bool operator()(const T* const b, T* e) const {
        const T fm = b[0];
//        T fm = T(0);
        const T r2star = b[1];
//        const double r2star = r2star;
//        float fm = 0;
//        T r2star = T(0);

        for (int j = 0; j < NRESIDUALS; j++) {
            complext<T> predicted = complext<T>(T(0.0));
            const float TE = TEs_[j];
            for (int i = 0; i < NSPECIES; i++) {
                auto &species = alg_.species_[i];
                complext<T> tmp(b[2*i+2],b[2*i+3]);
//                complext<T> tmp(b[2*i+2]);
//                T tmp = b[i+2];
                predicted += tmp * std::accumulate(species.ampFreq_.begin(), species.ampFreq_.end(), complext<T>(T(0.0)),
                                                   [&](auto val, auto peak) {
                                                       return val + T(peak.first) * exp(complext<T>(T(0.0),T(2.0)) * T(PI * peak.second*fieldstrength_*GAMMABAR * TE));
                                                   });
            }
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


};




void Gadgetron::fat_water_mixed_fitting(hoNDArray<float> &field_map, hoNDArray<float> &r2star_map,
                                        hoNDArray<std::complex<float>> &fractions,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const FatWaterAlgorithm &alg_, const std::vector<float> &TEs, float fieldstrength){




    const size_t X = input_data.get_size(0);
    const size_t Y = input_data.get_size(1);
    const size_t N = input_data.get_size(4);
    const size_t S = input_data.get_size(5);

    std::vector<float> TEs_repeated((S-1)*N);
    std::vector<float> TEs_repeated1(N,TEs[0]);





    for (int k3 = 1; k3 < S; k3++) {
        for (int k4 = 0; k4 < N; k4++) {
            TEs_repeated[k4 + (k3-1) * N] = float(TEs[k3]);
        }
    }


#pragma omp parallel for collapse(2)
    for (int k2 = 0; k2 < Y; k2++){
        for (int k1 = 0; k1 < X; k1++){
            std::vector<complext<double>> signal1(N);
            std::vector<complext<double>> signal((S-1)*N);
            auto& f = field_map(k1,k2);
            auto& r2 = r2star_map(k1,k2);
            std::complex<float>& water = fractions(k1,k2,0,0,0,0,0);
            std::complex<float>& fat =   fractions(k1,k2,0,0,0,1,0);
            for (int k4 = 0; k4 < N; k4++) {
                signal1[k4] = input_data(k1, k2, 0, 0, k4, 0, 0);
            }
            for (int k3 = 1; k3 < S; k3++) {
                for (int k4 = 0; k4 < N; k4++) {
                    signal[k4+(k3-1)*N] = input_data(k1, k2, 0, 0, k4, k3, 0);
                }
            }

            ceres::Problem problem;
            /*
            auto cost_function1 = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,1,abs_residual>,ceres::RIDDERS,1,6>(
                    new FatWaterModelCeres<2,1,abs_residual>(alg_,TEs_repeated1,signal1,fieldstrength));



            auto cost_function = new ceres::NumericDiffCostFunction<FatWaterModelCeres<2,3,complex_residual>,ceres::RIDDERS,6,6>(
                    new FatWaterModelCeres<2,3,complex_residual>(alg_,TEs_repeated,signal,fieldstrength));
                    */


            auto cost_function1 = new ceres::AutoDiffCostFunction<FatWaterModelCeres<2,1,abs_residual>,1,6>(
                    new FatWaterModelCeres<2,1,abs_residual>(alg_,TEs_repeated1,signal1,fieldstrength,r2));



            auto cost_function = new ceres::AutoDiffCostFunction<FatWaterModelCeres<2,3,complex_residual>,6,6>(
                    new FatWaterModelCeres<2,3,complex_residual>(alg_,TEs_repeated,signal,fieldstrength,r2));

            std::vector<double> b = {f,r2,water.real(),water.imag(),fat.real(),fat.imag()};

            problem.AddResidualBlock(cost_function1, nullptr, b.data());
            problem.AddResidualBlock(cost_function, nullptr, b.data());
//            problem.SetParameterLowerBound(b.data(),1,0);
//            problem.SetParameterUpperBound(b.data(),1,200);

            ceres::Solver::Options options;
           options.max_num_iterations = 50;
            options.linear_solver_type = ceres::DENSE_QR;
//            options.initial_trust_region_radius = 0.1;
//            options.dense_linear_algebra_library_type = ceres::LAPACK;
//            options.use_inner_iterations = true;

//        options.use_explicit_schur_complement = true;
//            options.function_tolerance = 1e-4;
//            options.gradient_tolerance = 1e-4;
//            options.parameter_tolerance = 1e-4;
//        options.preconditioner_type = ceres::IDENTITY;
//    options.minimizer_type = ceres::LINE_SEARCH;
//         options.line_search_direction_type = ceres::BFGS;
//        options.trust_region_strategy_type = ceres::DOGLEG;
//    options.dogleg_type = ceres::SUBSPACE_DOGLEG;


            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);



            f = b[0];
            r2 = b[1];
            water = std::complex<float>(b[2],b[3]);
            fat = std::complex<float>(b[4],b[5]);

        }
    }



}




