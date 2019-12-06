/** \file cuSenseOperator.h
    \brief Base class for the GPU based Sense operators
*/

#pragma once
#include "gpu_sense_utilities.h"
#include "../senseOperator.h"
#include "complext.h"
#include "cuNDArray_blas.h"
#include "cuNDArray_elemwise.h"
#include "cuNDArray_operators.h"
#include "gpupmri_export.h"
#include "vector_td.h"

namespace Gadgetron{
    template<class REAL, unsigned int D>
    using cuSenseOperator = senseOperator<cuNDArray<complext<REAL>>,D>;
}
