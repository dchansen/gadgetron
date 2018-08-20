#pragma once

#include "fatwater.h"
namespace Gadgetron { namespace FatWater {
    void field_map_ncg(hoNDArray<float> &field_map, const hoNDArray<float> &r2star_map,
                                        const hoNDArray<std::complex<float>> &input_data,
                                        const Parameters &parameters, float regularization_strength);

}}
