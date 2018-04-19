#pragma once

#include <cpu/hoNDArray.h>
#include "fatwater.h"

namespace Gadgetron {

    void spectral_separation_mixed_fitting(hoNDArray<float>& field_map, hoNDArray<float>& r2star_map,
            hoNDArray<std::complex<float>>& fractions,
    const hoNDArray<std::complex<float>>& input_data,
    const FatWaterAlgorithm& alg_, const std::vector<float>& TEs);

}