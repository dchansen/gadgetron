

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


            FieldMapModel(const hoNDArray<std::complex<float>> &measurements, const hoNDArray<float>& r2star) {

                angles = calculate_angles(measurements);
                weights = calculate_weights(measurements,r2star);

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
                                  for (int t2 = 0; t2 < times.size(); t2++) {
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
                  return total;

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
                            auto data_norm = 0;

                            for (int dt = 0; dt < measurement.get_size(3); dt++)
                                data_norm += norm(measurement(dx, dy, dz, dt)*r2star(dx,dy,dz));

                            for (int dt2 = 0; dt2 < measurement.get_size(3); dt2++) {
                                auto val2 = measurement(dx, dy, dz, dt2)*r2star(dx,dy,dz);
                                for (int dt1 = 0; dt1 < measurement.get_size(3); dt1++) {
                                    auto val1 = measurement(dx, dy, dz, dt1)*r2star(dx,dy,dz);

                                    result(dt1, dt2, dx, dy, dz) = norm(val1 * val2) / data_norm;
                                }
                            }
                        }
                    }
                }
                return result;
            }

            static float
            gradient_internal(float field_value, float time1, float time2, float angle1, float angle2, float weight) {
                return weight * (time1 - time2) * std::sin(angle1 - angle2 + field_value * (time1 - time2));
            }

            static float
            surrogate_d_internal(float field_value, float time1, float time2, float angle1, float angle2, float weight){
                float s = fmod(field_value*(time1-time2)+angle1-angle2, PI);
                float time_diff = time1-time2;
                return weight*time_diff*time_diff*std::sin(s)/s;

            }


            hoNDArray<float> weights;
            hoNDArray<float> angles;
            std::vector<float> times;
        };


        float step_size(const FieldMapModel& model, const hoNDArray<float>& field_map,const hoNDArray<float>& step_direction, const hoNDArray<float>& model_gradient,
                const hoNDArray<float>& regularization_step_gradient, float regularization_strength){

            return sum(&model_gradient)/(model.surrogate_d(field_map,step_direction)+regularization_strength*dot(&regularization_step_gradient,&step_direction));

        }


        }
        void field_map_ncg(hoNDArray<float> &field_map, const hoNDArray<float> &r2star_map,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const Parameters &parameters, float regularization_strength) {



                const auto model = FieldMapModel(input_data,r2star_map);

                std::vector<hoPartialDerivativeOperator<float,3>> finite_difference =
                        {hoPartialDerivativeOperator<float,3>(0),hoPartialDerivativeOperator<float,3>(1)};

                if (input_data.get_size(2) > 1)
                    finite_difference.push_back(hoPartialDerivativeOperator<float,3>(2));

                hoNDArray<float> model_gradient = model.gradient(field_map);
                hoNDArray<float> regularization_gradient(field_map.get_dimensions());
                regularization_gradient.fill(0);
                for (auto& op : finite_difference){
                    op.mult_MH_M(&field_map,&regularization_gradient,true);
                }
                regularization_gradient *= regularization_strength;

                hoNDArray<float> step_direction(model_gradient.get_dimensions());

                axpy(regularization_strength,regularization_gradient,model_gradient,step_direction);


                hoNDArray<float> regularization_gradient_update(field_map.get_dimensions());
                regularization_gradient_update.fill(0);
                 for (auto& op : finite_difference){
                    op.mult_MH_M(&step_direction,&regularization_gradient_update,true);
                }

                float alpha = step_size(model, field_map,step_direction,model_gradient,regularization_gradient_update,regularization_strength);

                 axpy(alpha,regularization_gradient_update,regularization_gradient,regularization_gradient);

                 axpy(alpha,step_direction,field_map,field_map);


                 auto previous_gradient = step_direction;

                for (int i = 0; i < 10; i++){
                    model_gradient = model.gradient(field_map);

                    regularization_gradient *= regularization_strength;
                    auto combined_gradient = model_gradient;
                    axpy(regularization_strength,regularization_gradient,combined_gradient,combined_gradient);
                    auto denom = dot(&previous_gradient,&previous_gradient);
                    previous_gradient -= combined_gradient;
                    auto update = dot(&previous_gradient,&combined_gradient)/denom;
                    previous_gradient = combined_gradient;

                    step_direction *= update;
                    step_direction += combined_gradient;

                    regularization_gradient_update.fill(0);
                    for (auto& op : finite_difference){
                        op.mult_MH_M(&step_direction,&regularization_gradient_update,true);
                    }

                    float alpha = step_size(model, field_map,step_direction,model_gradient,regularization_gradient_update,regularization_strength);

                    axpy(alpha,regularization_gradient_update,regularization_gradient,regularization_gradient);
                    axpy(alpha,step_direction,field_map,field_map);

                }
        };



    }
}
