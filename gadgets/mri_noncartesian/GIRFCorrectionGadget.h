#pragma once

#include "Gadget.h"
#include <ismrmrd/ismrmrd.h>
#include "hoNDArray.h"
namespace Gadgetron {

  class GIRFCorrectionGadget :
    public Gadget2<ISMRMRD::AcquisitionHeader,hoNDArray< std::complex<float> > >
    {
    public:
      GADGET_DECLARE(NoiseAdjustGadget);


      GIRFGadget();
      virtual ~GIRFGadget();


    protected:
      GADGET_PROPERTY(girf_dependency_prefix, std::string, "Prefix of noise depencency file", "GadgetronNoiseCovarianceMatrix");
      GADGET_PROPERTY(perform_girf_correction, bool, "Whether to actually perform the noise adjust", true);
      GADGET_PROPERTY(pass_nonconformant_data, bool, "Whether to pass data that does not conform", false);


    };
}



