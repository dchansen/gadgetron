#pragma once

#include "Node.h"
#include "hoNDArray.h"

#include "mri_core_acquisition_bucket.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <map>

namespace Gadgetron {



    class AccumulateFlagTrigger
        : public Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>> {
    public:
        using Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>::ChannelGadget;
        void process(Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>>& in,
            Core::OutputChannel& out) override;
        enum class TriggerFlag {
            ENCODE_STEP1,
            ENCODE_STEP2,
            AVERAGE,
            SLICE,
            CONTRAST,
            PHASE,
            REPETITION,
            SET,
            SEGMENT,
            MEASUREMENT

        };
        NODE_PROPERTY(trigger_flag, TriggerFlag, "Dimension to trigger on", TriggerDimension::none);

    private:
        void send_data(Core::OutputChannel& out, std::map<unsigned short, AcquisitionBucket>& buckets,
                       std::vector<Core::Waveform>& waveforms);
    };

    void from_string(const std::string& str, AcquisitionAccumulateTriggerGadget::TriggerDimension& val);

}
