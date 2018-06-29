#pragma once

#include "hoNDArray.h"
#include "fatwater.h"

namespace Gadgetron{
    namespace FatWater {
                void field_map_fitting(hoNDArray<float> &field_map, const hoNDArray<float> &r2star_map,
                               const hoNDArray<std::complex<float>> &input_data,
                               const hoNDArray<float> &lambda_map, const Parameters &parameters );
    }
}
