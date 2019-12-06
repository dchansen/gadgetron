#pragma once

#include "cuNDArray.h"
#include "complext.h"
#include "gpupmri_export.h"

namespace Gadgetron{
    namespace Sense {

    // Multiply with coil sensitivities
    //

    template <class REAL, unsigned int D>
    void csm_mult_M(
        const cuNDArray<complext<REAL>> &in, cuNDArray<complext<REAL>> &out, const cuNDArray<complext<REAL>> &csm);

    // Multiply with adjoint of coil sensitivities
    //

    template <class REAL, unsigned int D>
    void csm_mult_MH(
        const cuNDArray<complext<REAL>> &in, cuNDArray<complext<REAL>> &out, const cuNDArray<complext<REAL>> &csm);
}
}
