#pragma once

#include "hoNDArray.h"
#include <ceres/ceres.h>

namespace Gadgetron {
    namespace FatWater {
        namespace {


            struct DiffLoss {
                DiffLoss(double scale1) : scale1_(scale1) {}

                template<class T>
                bool operator()(const T *const base, const T *const dx, T *residual) const {

                    residual[0] = scale1_ * (base[0] - dx[0]);

                    return true;
                }

            private:
                const double scale1_;

            };

            void
            add_regularization(ceres::Problem &problem, hoNDArray<double> &field_map,
                               const hoNDArray<float> &lambda_map,
                               ceres::LossFunction *loss = NULL) {


                auto add_term = [&](int x1, int y1, int z1, int x2, int y2, int z2) {
                    auto weight = std::min(lambda_map(x1, y1, z1), lambda_map(x2, y2, z2));
                    auto cost_function = new ceres::AutoDiffCostFunction<DiffLoss, 1, 1, 1>(new DiffLoss(weight));
                    std::vector<double *> ptrs = {&field_map(x1, y1, z1), &field_map(x2, y2, z2)};
                    problem.AddResidualBlock(cost_function, loss, ptrs);
                };
                const size_t X = field_map.get_size(0);
                const size_t Y = field_map.get_size(1);
                const size_t Z = field_map.get_size(2);

                for (int kz = 0; kz < Z; kz++) {
                    for (int ky = 0; ky < Y; ky++) {
                        for (int kx = 0; kx < X; kx++) {

                            if (kx < X - 1) {
                                add_term(kx, ky, kz, kx + 1, ky, kz);
                            }
                            if (ky < Y - 1) {
                                add_term(kx, ky, kz, kx, ky + 1, kz);
                            }
                            if (kz < Z - 1) {
                                add_term(kx, ky, kz, kx, ky, kz + 1);
                            }

                        }
                    }
                }
            }
        }
    }
}

