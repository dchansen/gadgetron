#pragma once

#include "Gadget.h"
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/waveform.h>
#include "hoNDArray.h"
namespace Gadgetron {

  class GIRFCorrectionGadget :
    public BasicPropertyGadget
    {
    public:
      GADGET_DECLARE(GIRFCorrectionGadget);


      virtual ~GIRFCorrectionGadget();




    protected:
      GADGET_PROPERTY(girf_dependency_prefix, std::string, "Prefix of noise depencency file", "GadgetronNoiseCovarianceMatrix");
      GADGET_PROPERTY(perform_girf_correction, bool, "Whether to actually perform the noise adjust", true);
      GADGET_PROPERTY(pass_nonconformant_data, bool, "Whether to pass data that does not conform", false);

      virtual int process_config(ACE_Message_Block* mb) override;

      int process(GadgetContainerMessage<ISMRMRD::WaveformHeader>* m1, GadgetContainerMessage<hoNDArray<uint32_t>>* m2);
      int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader>* m1,
	  GadgetContainerMessage< hoNDArray< std::complex<float> > >* m2);

      static void save_girf(const ISMRMRD::AcquisitionHeader& header,const hoNDArray<std::complex<float>> data);

    };
}



