//
// Created by dchansen on 5/14/18.
//

#pragma  once

#include <cpu/hoNDArray.h>

namespace Gadgetron {

    template<unsigned int N>
    void fatwaterICM(hoNDArray <uint16_t> &field_map_index, const hoNDArray<float> &residual_map,
                         const std::vector<float> &field_map_strengths, const int iterations, const float sigma);

}



