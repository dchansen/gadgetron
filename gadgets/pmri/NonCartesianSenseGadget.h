//
// Created by david on 27/11/2019.
//

#pragma once
#include "PureGadget.h"
#include "mri_core_data.h"
#include "GenericReconJob.h"
#include "cgSolver.h"
namespace Gadgetron {
    template<class ARRAY>
    class NonCartesianSenseGadget :  Core::PureGadget<IsmrmrdImageArray, GenericReconJob> {
        IsmrmrdImageArray process_function(GenericReconJob args) const override {

            cgSolver<ARRAY> solver;
            auto encoding_operator = boost::make_shared<NFFTOperator>

        }

    };
}


