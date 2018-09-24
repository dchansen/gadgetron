//
// Created by dchansen on 9/18/18.
//

#include "GIRFCorrectionGadget.h"

namespace Gadgetron {
    GIRFCorrectionGadget::~GIRFCorrectionGadget() {

    }


    int GIRFCorrectionGadget::process_config(ACE_Message_Block *mb) {
        return Gadget::process_config(mb);
    }

    int GIRFCorrectionGadget::process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
                                                 GadgetContainerMessage<hoNDArray<std::complex<float>>> *m2) {
        auto& header = *m1->getObjectPtr();
        bool is_girf = header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_GIRF_MEASUREMENT);
        if (is_girf){
            save_girf(header,*m2->getObjectPtr());
        }

        if (pass_on_undesired_data)
            this->put_next(m1);

        return 0;
    }

    void GIRFCorrectionGadget::save_girf(const ISMRMRD::AcquisitionHeader &header,
                                         const hoNDArray<std::complex<float>> data) {

        throw std::runtime_error("We should really implement this");
    }

    int GIRFCorrectionGadget::process(GadgetContainerMessage<ISMRMRD::WaveformHeader> *m1,
                                      GadgetContainerMessage<hoNDArray<uint32_t>> *m2) {
        return 0;
    }


    




}