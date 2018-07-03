
#include "FieldMapFitting.h"
#include <complex>
#include <numeric>
#include <ceres/ceres.h>
#include <boost/math/constants/constants.hpp>
#include "fitting_utilities.h"

constexpr double PI = boost::math::constants::pi<double>();
namespace Gadgetron {
    namespace FatWater {

        namespace {
            template<unsigned int N>
            struct FieldMapModel {

                FieldMapModel(const FatWater::Parameters &parameters, const std::vector<float> &TEs,
                              const std::vector<complext<double>> &signal, float r2star): TEs_(TEs) {


                    std::vector<complext<double>> data = signal;

                    for (int i = 0; i < data.size(); i++) {
                        data[i] *= std::exp(r2star * TEs[i]);
                    }

                    omega = parameters.gyromagnetic_ratio_Mhz*PI;


                    std::transform(data.begin(), data.end(), angles.begin(), [](auto c) { return arg(c); });


                    auto data_norm = std::accumulate(data.begin(), data.end(), 0.0,
                                                [](auto acc, auto c) { return acc + norm(c); });
                    for (int i = 0; i < data.size(); i++) {
                        for (int j = 0; j < data.size(); j++) {
                            weights[j + i * N] = norm(data[i] * data[j]) / data_norm;
                        }
                    }
                }


                template<class T>
                bool operator()(const T *const fm, T *residual) const {
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            residual[j + i * N] = sqrt(weights[j + i * N] * (1.0 - cos(*fm*omega * T(TEs_[i] - TEs_[j]) +
                                                                                       angles[i] + angles[j])));
                        }
                    }
                    return true;
                }

                const std::vector<float> TEs_;
                std::array<double, N*N> weights;
                std::array<double, N> angles;
                double omega;

            };


            template<unsigned int ECHOES>
            void field_map_fitting_echo(hoNDArray<float> &field_mapF, const hoNDArray<float> &r2star_mapF,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const hoNDArray<float> &lambda_map, const Parameters &parameters) {


                hoNDArray<double> field_map;
                field_map.copyFrom(field_mapF);
                hoNDArray<double> r2star_map;
                r2star_map.copyFrom(r2star_mapF);


                const size_t X = input_data.get_size(0);
                const size_t Y = input_data.get_size(1);
                const size_t Z = input_data.get_size(2);
                const size_t N = input_data.get_size(4);
                const size_t S = input_data.get_size(5);

                std::vector<float> TEs_repeated((S) * N);

                auto &TEs = parameters.echo_times_s;
                auto &field_strength = parameters.field_strength_T;

                for (int k3 = 0; k3 < S; k3++) {
                    for (int k4 = 0; k4 < N; k4++) {
                        TEs_repeated[k4 + (k3) * N] = double(TEs[k3]);
                    }
                }

                ceres::Problem problem;
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::CGNR;
                options.num_threads = omp_get_max_threads();
                options.use_inner_iterations = true;
                options.num_linear_solver_threads = omp_get_max_threads();
                std::cout << "Num threads: " << options.num_threads << std::endl;
                options.dense_linear_algebra_library_type = ceres::EIGEN;
                options.function_tolerance = 1e-6;
                options.gradient_tolerance = 1e-6;
                options.parameter_tolerance = 1e-6;


                for (int kz = 0; kz < Z; kz++) {
                    for (int ky = 0; ky < Y; ky++) {
                        for (int kx = 0; kx < X; kx++) {

                            std::vector<complext<double>> signal((S) * N);
                            auto &f = field_map(kx, ky, kz);
                            auto r2 = r2star_map(kx, ky, kz);

                            for (int k3 = 1; k3 < S; k3++) {
                                for (int k4 = 0; k4 < N; k4++) {
                                    signal[k4 + (k3) * N] = input_data(kx, ky, kz, 0, k4, k3, 0);
                                }
                            }

                            auto cost_function = new ceres::AutoDiffCostFunction<FieldMapModel<ECHOES>,
                                    ECHOES * ECHOES, 1>(
                                    new FieldMapModel<ECHOES>(parameters, TEs_repeated, signal, r2));

                            problem.AddResidualBlock(cost_function, nullptr, &f);
                        }
                    }
                }
                add_regularization(problem, field_map, lambda_map);
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);
                std::cout << "Initial cost: " << summary.initial_cost << " Final cost:" << summary.final_cost
                          << " Iterations "
                          << summary.iterations.size() << std::endl;

                field_mapF.copyFrom(field_map);


            }
        }

        void field_map_fitting(hoNDArray<float> &field_map, const hoNDArray<float> &r2star_map,
                               const hoNDArray<std::complex<float>> &input_data,
                               const hoNDArray<float> &lambda_map, const Parameters &parameters) {
            switch (parameters.echo_times_s.size()) {
                case 3:
                    field_map_fitting_echo<3>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                case 4:
                    field_map_fitting_echo<4>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                case 5:
                    field_map_fitting_echo<5>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                case 6:
                    field_map_fitting_echo<6>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                case 7:
                    field_map_fitting_echo<7>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                case 8:
                    field_map_fitting_echo<8>(field_map, r2star_map, input_data, lambda_map, parameters);
                    break;
                default:
                    throw std::invalid_argument("Fat water ftting only supported for 3 to 8 echoes");
            }
        }
    }
}
