//
// Created by dchansen on 12/6/19.
//

#pragma once
#include "complext.h"
#include "hoNDArray.h"

namespace Gadgetron{
namespace Sense {
    // Multiply with coil sensitivities
    //

    template <class REAL, unsigned int D>
    void csm_mult_M(const hoNDArray<complext<REAL>>& in, hoNDArray<complext<REAL>>& out, const hoNDArray<complext<REAL>>& csm);

    // Multiply with adjoint of coil sensitivities
    //

    template <class REAL, unsigned int D>
    void csm_mult_MH(const hoNDArray<complext<REAL>>& in, const hoNDArray<complext<REAL>>& out, const hoNDArray<complext<REAL>>& csm);


}
}
