/** \file hoNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once
#include "cpu_sense_utilities.h"
#include "NonCartesianSenseOperator.h"
#include "hoNFFT.h"
#include "senseOperator.h"
#include "hoNDArray_math.h"

namespace Gadgetron {

    template <class REAL, unsigned int D>
    class hoNonCartesianSenseOperator : public NonCartesianSenseOperator<hoNDArray, REAL, D> {

    public:
        using _uint64d = vector_td<size_t,D>;
        using _reald   = vector_td<REAL, D>;

        using NonCartesianSenseOperator<hoNDArray,REAL,D>::NonCartesianSenseOperator;
        virtual ~hoNonCartesianSenseOperator() = default;

        void setup(_uint64d matrix_size, _uint64d matrix_size_os, REAL W) override;
    };

}
