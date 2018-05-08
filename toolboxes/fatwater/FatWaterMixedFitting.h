#pragma once

#include <cpu/hoNDArray.h>
#include "fatwater.h"

namespace Gadgetron {

    void fat_water_mixed_fitting(hoNDArray<float> &field_map, hoNDArray<float> &r2star_map,
                                     hoNDArray <std::complex<float>> &fractions,
                                     const hoNDArray <std::complex<float>> &input_data,
                                     const hoNDArray<float> &lambda_map, const FatWaterAlgorithm &alg_,
                                     const std::vector<float> &TEs, float fieldstrength);

}