#include "hoNonCartesianSenseOperator.h"
#include "NonCartesianSenseOperator.h"
//
// Created by david on 16/12/2019.
//
namespace Gadgetron {
    template<class REAL, unsigned int D>
    void Gadgetron::hoNonCartesianSenseOperator<REAL, D>::setup(_uint64d matrix_size,
                                                                _uint64d matrix_size_os,
                                                                REAL W) {

        this->plan_ = NFFT<hoNDArray, REAL, D>::make_plan(matrix_size, matrix_size_os, W);
    }

    template
    class hoNonCartesianSenseOperator<float, 1>;

    template
    class hoNonCartesianSenseOperator<float, 2>;

    template
    class hoNonCartesianSenseOperator<float, 3>;

    template
    class hoNonCartesianSenseOperator<float, 4>;

    template
    class hoNonCartesianSenseOperator<double, 1>;

    template
    class hoNonCartesianSenseOperator<double, 2>;

    template
    class hoNonCartesianSenseOperator<double, 3>;

    template
    class hoNonCartesianSenseOperator<double, 4>;
}