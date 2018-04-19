//
// Created by dch on 18/04/18.
//

#include "FatWaterMixedFitting.h"
#include "fatwater.h"

#include <numeric>
#include <boost/math/constants/constants.hpp>
#include <limits>
namespace std {
    template<> class numeric_limits<std::complex<float>> : public numeric_limits<float>{


    };
}

#include <dlib/optimization.h>

using namespace Gadgetron;

using namespace std::complex_literals;
constexpr float PI = boost::math::constants::pi<float>();
template<unsigned int NSPECIES> class FatWaterModel {
public:
    typedef dlib::matrix<float,NSPECIES+2,1> parameter_vector;

    FatWaterModel(FatWaterAlgorithm alg) : alg_(alg){
        assert(alg_.species_.size() == NSPECIES);
    };

    std::complex<float> operator()(const parameter_vector& data, float TE ) {
        const float fieldstrength = data(0);
        const float r2star = data(1);

        std::complex<float> result = 0if;
        for (int i = 0; i < NSPECIES; i++){
            auto& species = alg_.species_[i];
            std::complex<float> tmp(data(i*2+2),data(i*2+3));
            result += tmp*std::accumulate(species.ampFreq_.begin(), species.ampFreq_.end(), 0if, [&](auto val, auto peak) {
               return val + peak.first * std::exp(2if * PI * peak.second * TE);
           });
        }

        result *=  std::exp((-r2star + 2if * fieldstrength) * TE);

        return result;
    }

private:
    const FatWaterAlgorithm alg_;


};

template<class T> class objective_delta_stop_strategy
{
public:
    explicit objective_delta_stop_strategy (
            double min_delta = 1e-7
    ) : _verbose(false), _been_used(false), _min_delta(min_delta), _max_iter(0), _cur_iter(0), _prev_funct_value(0)
    {
        DLIB_ASSERT (
                min_delta >= 0,
                "\t objective_delta_stop_strategy(min_delta)"
                        << "\n\t min_delta can't be negative"
                        << "\n\t min_delta: " << min_delta
        );
    }

    objective_delta_stop_strategy (
            double min_delta,
            unsigned long max_iter
    ) : _verbose(false), _been_used(false), _min_delta(min_delta), _max_iter(max_iter), _cur_iter(0), _prev_funct_value(0)
    {
        DLIB_ASSERT (
                min_delta >= 0 && max_iter > 0,
                "\t objective_delta_stop_strategy(min_delta, max_iter)"
                        << "\n\t min_delta can't be negative and max_iter can't be 0"
                        << "\n\t min_delta: " << min_delta
                        << "\n\t max_iter:  " << max_iter
        );
    }

    objective_delta_stop_strategy& be_verbose(
    )
    {
        _verbose = true;
        return *this;
    }


    bool should_continue_search (
            const T& ,
            const T funct_value,
            const T&
    )
    {
        if (_verbose)
        {
            using namespace std;
            cout << "iteration: " << _cur_iter << "   objective: " << funct_value << endl;
        }

        ++_cur_iter;
        if (_been_used)
        {
            // Check if we have hit the max allowable number of iterations.  (but only
            // check if _max_iter is enabled (i.e. not 0)).
            if (_max_iter != 0 && _cur_iter > _max_iter)
                return false;

            // check if the function change was too small
            if (std::abs(funct_value - _prev_funct_value) < _min_delta)
                return false;
        }

        _been_used = true;
        _prev_funct_value = funct_value;
        return true;
    }

private:
    bool _verbose;

    bool _been_used;
    double _min_delta;
    unsigned long _max_iter;
    unsigned long _cur_iter;
    T _prev_funct_value;
};


void Gadgetron::spectral_separation_mixed_fitting(hoNDArray<float>& field_map, hoNDArray<float>& r2star_map,
                                                  hoNDArray<std::complex<float>>& fractions,
                                                  const hoNDArray<std::complex<float>>& input_data,
                                                  const FatWaterAlgorithm& alg_, const std::vector<float>& TEs){

    FatWaterModel<2> model(alg_);
    typedef FatWaterModel<2>::parameter_vector parameter_vector;

    const size_t X = input_data.get_size(0);
    const size_t Y = input_data.get_size(1);
    const size_t N = input_data.get_size(4);
    const size_t S = input_data.get_size(5);

    for (int k2 = 0; k2 < Y; k2++){
        for (int k1 = 0; k1 < X; k1++){
            const auto f = field_map(k1,k2);
            const auto r2 = r2star_map(k1,k2);
            const auto water = fractions(0,k1,k2);
            const auto fat = fractions(1,k1,k2);
            std::vector<std::pair<float, std::complex<float>>> fitting_data(S);
            for (int k3 = 0; k3 < S; k3++) {
                for (int k4 = 0; k4 < N; k4++) {
                    fitting_data[k4 + k3 * N] = std::make_pair(float(TEs[k3]), std::complex<float>(
                            input_data(k1, k2, 0, 0, k4, k3, 0)));
                }
            }

            auto residual = [&](const std::pair<float,std::complex<float>>& data,const parameter_vector& params ){
                return std::abs(model(params,data.first)-data.second);
            };
            parameter_vector x = {f,r2,water,fat};

            dlib::solve_least_squares_lm(dlib::objective_delta_stop_strategy(1e-4),
                                         residual,
                                         dlib::derivative(residual),
                                         fitting_data,
                                         x);

            field_map(k1,k2) = x(0).real();
            r2star_map(k1,k2) = x(1).real();
            fractions(0,k1,k2) = x(2);
            fractions(1,k1,k2) = x(3);



        }
    }


}




