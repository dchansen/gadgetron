

//

#include "optimization_transfer.h"
#include "hoNDArray.h"
#include <boost/math/constants/constants.hpp>
#include "hoPartialDerivativeOperator.h"

//Implements a nonlinear conjugate gradient algorithm for field map estimation, based on
//G. Ongie, J. Shi and J. A. Fessler, "Efficient computation of regularized field map estimates in 3D,"
// 2017 IEEE 14th International Symposium on Biomedical Imaging (ISBI 2017), Melbourne, VIC, 2017, pp. 700-703.
//doi: 10.1109/ISBI.2017.7950616


namespace Gadgetron {
    namespace FatWater {

        namespace {

        constexpr float PI = boost::math::constants::pi<float>();

        class FieldMapModel {

        public:


            FieldMapModel(const hoNDArray<std::complex<float>> &measurements, const hoNDArray<float>& r2star, const std::vector<float>& times) {

                angles = calculate_angles(measurements);
                weights = calculate_weights(measurements,r2star);
                this->times = times;

            }

            float magnitude(const hoNDArray<float> &field_map) const {

                float total = 0;

#pragma omp parallel for collapse(3) reduction (+ : total)
                  for (int dz = 0; dz < field_map.get_size(2); dz++) {
                      for (int dy = 0; dy < field_map.get_size(1); dy++) {
                          for (int dx = 0; dx < field_map.get_size(0); dx++) {
                              for (int t1 = 0; t1 < times.size(); t1++) {
                                  for (int t2 = 0; t2 < times.size(); t2++) {
                                      total += magnitude_internal(field_map(dx, dy, dz), times[t1], times[t2],
                                                                  angles(t1, dx, dy, dz),
                                                                  angles(t2, dx, dy, dz),
                                                                  weights(t1, t2, dx, dy, dz));
                                  }

                              }
                          }
                      }
                  }
                  return total;

            }


            hoNDArray<float> gradient(const hoNDArray<float> &field_map) const {

                hoNDArray<float> result(field_map.get_dimensions());
                const size_t elements = field_map.get_number_of_elements();

#pragma omp parallel for collapse(3)
                for (int dz = 0; dz < field_map.get_size(2); dz++)
                    for (int dy = 0; dy < field_map.get_size(1); dy++)
                        for (int dx = 0; dx < field_map.get_size(0); dx++) {
                            result(dx, dy, dz) = 0;
                            for (int t1 = 0; t1 < times.size(); t1++) {
                                for (int t2 = 0; t2 < times.size(); t2++) {
                                    result(dx, dy, dz) += gradient_internal(field_map(dx, dy, dz), times[t1], times[t2],
                                                                            angles(t1, dx, dy, dz),
                                                                            angles(t2, dx, dy, dz),
                                                                            weights(t1, t2, dx, dy, dz));
                                }

                            }
                        }
                return result;
            }


            float surrogate_d(const hoNDArray<float>& field_map, const hoNDArray<float>& step_direction) const {

                float total = 0;

#pragma omp parallel for collapse(3) reduction (+ : total)
                  for (int dz = 0; dz < step_direction.get_size(2); dz++) {
                      for (int dy = 0; dy < step_direction.get_size(1); dy++) {
                          for (int dx = 0; dx < step_direction.get_size(0); dx++) {
                              for (int t1 = 0; t1 < times.size(); t1++) {
                                  for (int t2 = t1+1; t2 < times.size(); t2++) {
                                      total += surrogate_d_internal(field_map(dx, dy, dz), times[t1], times[t2],
                                                                    angles(t1, dx, dy, dz),
                                                                    angles(t2, dx, dy, dz),
                                                                    weights(t1, t2, dx, dy, dz))*
                                                                            norm(step_direction(dx,dy,dz));
                                  }

                              }
                          }
                      }
                  }
                  return 2*total;

            }


        private:


            static hoNDArray<float> calculate_angles(const hoNDArray<std::complex<float>> &measurement) {

                hoNDArray<float> result(measurement.get_size(3), measurement.get_size(0), measurement.get_size(1),
                                        measurement.get_size(2));

                for (int dz = 0; dz < measurement.get_size(2); dz++) {
                    for (int dy = 0; dy < measurement.get_size(1); dy++) {
                        for (int dx = 0; dx < measurement.get_size(0); dx++) {
                            for (int dt = 0; dt < measurement.get_size(3); dt++) {
                                result(dt, dx, dy, dz) = std::arg(measurement(dx, dy, dz, dt));
                            }
                        }
                    }
                }
                return result;
            }

            static hoNDArray<float> calculate_weights(const hoNDArray<std::complex<float>> &measurement, const hoNDArray<float>& r2star) {

                hoNDArray<float> result(measurement.get_size(3), measurement.get_size(3), measurement.get_size(0),
                                        measurement.get_size(1), measurement.get_size(2));

                for (int dz = 0; dz < measurement.get_size(2); dz++) {
                    for (int dy = 0; dy < measurement.get_size(1); dy++) {
                        for (int dx = 0; dx < measurement.get_size(0); dx++) {
                            float data_norm = 0;

                            for (int dt = 0; dt < measurement.get_size(3); dt++)
                                data_norm += std::norm(measurement(dx, dy, dz, dt));

                            for (int dt2 = 0; dt2 < measurement.get_size(3); dt2++) {
                                auto val2 = measurement(dx, dy, dz, dt2);
                                for (int dt1 = 0;  dt1 < measurement.get_size(3); dt1++) {
                                    auto val1 = measurement(dx, dy, dz, dt1);

                                    result(dt1, dt2, dx, dy, dz) = std::norm(val1 * val2) / data_norm;
                                    assert(result(dt1,dt2,dx,dy,dz) >= 0);
                                }
                            }
                        }
                    }
                }
                return result;
            }

            float
            gradient_internal(float field_value, float time1, float time2, float angle1, float angle2, float weight) const {
                return weight * (time1 - time2) * std::sin(angle1 - angle2 + field_value* (time1 - time2));
            }

            float
            surrogate_d_internal(float field_value, float time1, float time2, float angle1, float angle2, float weight) const {
                float s = remainder(field_value*(time1-time2)+angle1-angle2, PI);
                float sins;
                if (std::abs(s) < 1e-6) sins = 1;
                else sins = std::sin(s)/s;
                float time_diff = time1-time2;
                float result = weight*time_diff*time_diff*sins;
                assert(!std::isnan(result));
                return result;

            }

            float magnitude_internal(float field_value, float time1, float time2, float angle1, float angle2,
                                            float weight) const {
                assert(weight >= 0);
                return weight*(1.0f-std::cos(field_value*(time1-time2)+angle1-angle2));

            }


            hoNDArray<float> weights;
            hoNDArray<float> angles;
            std::vector<float> times;
        };


        float make_step(const FieldMapModel& model, hoNDArray<float>& field_map,const hoNDArray<float>& step_direction, hoNDArray<float>& model_gradient,
                const hoNDArray<float>& regularization_step_gradient, float regularization_strength){


            float alpha = 0;

            float reg = regularization_strength * dot(&regularization_step_gradient, &step_direction);
            for (int i = 0; i < 5; i++) {
                model_gradient = model.gradient(field_map);
                float surr = model.surrogate_d(field_map, step_direction);
                assert(!std::isnan(surr));

                assert(!std::isinf(surr));
                float step = -sum(&model_gradient)/(surr+reg);

                assert(!std::isinf(step));
                assert(!std::isnan(step));
                axpy(step,step_direction,field_map,field_map);
                alpha += step;

            }
            std::cout << "Alpha " << alpha << std::endl;
            return alpha;
        }


        hoNDArray<float> calc_regularization_gradient(std::vector<hoPartialDerivativeOperator<float,3>>& finite_difference, hoNDArray<float>& field_map){
            hoNDArray<float> result(field_map.get_dimensions());
            result.fill(0);
            for (auto& op : finite_difference)
                op.mult_MH_M(&field_map,&result,true);

            return result;
        }


        }
        void field_map_ncg(hoNDArray<float> &field_map, const hoNDArray<float> &r2star_map,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const Parameters &parameters, float regularization_strength) {

                const size_t X = input_data.get_size(0);
                const size_t Y = input_data.get_size(1);
                const size_t Z = input_data.get_size(2);
                const size_t N = input_data.get_size(4);
                if (N != 1)
                    throw std::runtime_error("Unsupported dimensions");
                const size_t S = input_data.get_size(5);

                hoNDArray<std::complex<float>> input_data_view({X,Y,Z,S},input_data.get_data_ptr());


                const auto model = FieldMapModel(input_data_view,r2star_map,parameters.echo_times_s);

                std::vector<hoPartialDerivativeOperator<float,3>> finite_difference =
                        {hoPartialDerivativeOperator<float,3>(0),hoPartialDerivativeOperator<float,3>(1)};

                if (input_data.get_size(2) > 1)
                    finite_difference.push_back(hoPartialDerivativeOperator<float,3>(2));

                for (auto& op: finite_difference) {
                    op.set_domain_dimensions(field_map.get_dimensions().get());
                    op.set_codomain_dimensions(field_map.get_dimensions().get());
                }

                hoNDArray<float> model_gradient = model.gradient(field_map);
                hoNDArray<float> regularization_gradient = calc_regularization_gradient(finite_difference,field_map);

                hoNDArray<float> step_direction(model_gradient.get_dimensions());
                step_direction.fill(0);

                axpy(regularization_strength,regularization_gradient,model_gradient,step_direction);

                hoNDArray<float> regularization_gradient_update = calc_regularization_gradient(finite_difference,step_direction);

                float alpha = make_step(model, field_map,step_direction,model_gradient,regularization_gradient_update,regularization_strength);


                 axpy(alpha,regularization_gradient_update,regularization_gradient,regularization_gradient);

                 auto previous_gradient = step_direction;

                for (int i = 0; i < 50; i++){

                    auto combined_gradient = model_gradient;
                    axpy(regularization_strength,regularization_gradient,combined_gradient,combined_gradient);
                    auto denom = dot(&previous_gradient,&previous_gradient);
                    previous_gradient -= combined_gradient;
                    auto update = -dot(&previous_gradient,&combined_gradient)/denom;
                    previous_gradient = combined_gradient;

                    update = std::max(update,0.0f);
                    step_direction *= update;
                    step_direction += combined_gradient;

                    regularization_gradient_update = calc_regularization_gradient(finite_difference,step_direction);

                    float alpha = make_step(model, field_map,step_direction,model_gradient,regularization_gradient_update,regularization_strength);

                    axpy(alpha,regularization_gradient_update,regularization_gradient,regularization_gradient);


                    float value = model.magnitude(field_map);
                    std::cout << "Model cost " << value << std::endl;
                    for (auto& op : finite_difference){
                        value += regularization_strength*op.magnitude(&field_map);
                    }

                    std::cout << "Cost " << value << " alpha " << alpha << std::endl;

                }
        };



    }
}
