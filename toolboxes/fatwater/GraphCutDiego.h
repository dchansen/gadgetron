#pragma once



#include "hoNDArray.h"

namespace Gadgetron {
    hoNDArray <uint16_t> doGraphCut(const hoNDArray <uint16_t> &cur_ind, const hoNDArray <uint16_t> &next_ind,
                                    const hoNDArray<float> &residual, const hoNDArray<float> &lmap, int size_clique);
}