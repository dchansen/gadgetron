//
// Created by dchansen on 5/14/18.
//

#include "ICM.h"
#include <random>
#include <boost/math/constants/constants.hpp>

#include <boost/range/numeric.hpp>
#include <boost/range/combine.hpp>
#include <boost/iterator/counting_iterator.hpp>

using namespace Gadgetron;
    static constexpr float PI = boost::math::constants::pi<float>();



    template <unsigned  int N> static std::array<float,N*N> load_values(int k1, int k2,const hoNDArray<uint16_t> &field_map_index, const std::vector<float> &field_map_strengths, size_t X, size_t Y ){

        std::array<float,N*N> values;
        values.fill(field_map_index(k1,k2));

        auto start1 = ((k1-int(N/2)) < 0) ? N/2-k1 : 0;
        auto start2 = ((k2-int(N/2)) < 0) ? N/2-k2 : 0;

        auto end1 = ((k1+int(N/2)) >= X) ? X-k1+N/2 : N;
        auto end2 = ((k2+int(N/2)) >= X) ? Y-k2+N/2 : N;

        for (int v2 = start2; v2 < end2; v2++) {
            for (int v1 = start1; v1 < end1; v1++) {
                values[v1+v2*N] = field_map_strengths[field_map_index(v1+k1-N/2,v2+k2-N/2)];
            }
        }

        return values;

    };


    template<unsigned int N> static uint16_t find_new_value(const std::array<float,N*N>& values, const std::array<float, N*N> kernel, const std::vector<float> residuals, const std::vector<float> &field_map_strengths, const float lambda){
        float min_cost = std::numeric_limits<float>::max();
        uint16_t  min_index = 0;
        for (int i = 0; i < residuals.size(); i++){
            auto field_value = field_map_strengths[i];
            auto cost = lambda*boost::accumulate(boost::combine(values,kernel),0, [&](auto val, auto tup){return std::pow((boost::get<0>(tup)-field_value)*boost::get<1>(tup),2)+val;})+residuals[i];
            if (cost < min_cost){
                min_cost = cost;
                min_index = i;
            }
        }
        return min_index;

    }
    template<unsigned int N>
    void Gadgetron::fatwaterICM(hoNDArray <uint16_t> &field_map_index, const hoNDArray<float> &residual_map,
                                    const std::vector<float> &field_map_strengths, const int iterations, const float lambda) {

        std::array<float, N * N> kernel;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                kernel[i + j * N] =
                        std::exp(-0.5 * (std::pow(i - N / 2, 2) + std::pow(j - N / 2, 2)))  /
                        (std::sqrt(2 * PI));
            }
        }
        kernel[N/2+(N/2)*N] = 0;
        std::mt19937 g(4242);


        const size_t X = field_map_index.get_size(0);
        const size_t Y = field_map_index.get_size(1);
        const size_t fm_sizes = residual_map.get_size(0);

        std::vector<int> y_index(boost::make_counting_iterator(size_t(0)),boost::make_counting_iterator(Y));
        std::vector<int> x_index(boost::make_counting_iterator(size_t(0)),boost::make_counting_iterator(X));

        for (int it = 0; it < iterations; it++) {
            std::shuffle(y_index.begin(),y_index.end(),g);

            /*
            if (it%2 ==0) {
                std::reverse(y_index.begin(), y_index.end());
            } else {
                std::reverse(x_index.begin(), x_index.end());
            }
*/

            for (auto k2 : y_index) {
                std::shuffle(x_index.begin(),x_index.end(),g);
                for (auto k1 : x_index) {

                    std::array<float, N * N> values = load_values<N>(k1, k2, field_map_index, field_map_strengths, X,
                                                                     Y);
                    auto residuals = std::vector<float>(&residual_map(0, k1, k2),
                                                        &residual_map(fm_sizes - 1, k1, k2) + 1);

                    field_map_index(k1, k2) = find_new_value<N>(values, kernel, residuals, field_map_strengths, lambda);

                }
            }
        }



    }

template  void Gadgetron::fatwaterICM<5>(hoNDArray <uint16_t> &field_map_index, const hoNDArray<float> &residual_map,
                            const std::vector<float> &field_map_strengths, const int iterations, const float sigma);