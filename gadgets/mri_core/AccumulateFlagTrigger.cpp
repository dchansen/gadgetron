#include "AccumulateFlagTrigger.h"
#include "log.h"
#include "mri_core_data.h"
#include <boost/algorithm/string.hpp>

namespace Gadgetron {
    using TriggerFlag = AccumulateFlagTrigger::TriggerDimension;
    namespace {
        bool is_noise(Core::Acquisition& acq) {
            return std::get<ISMRMRD::AcquisitionHeader>(acq).isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
        }


        void add_stats(AcquisitionBucketStats& stats, const ISMRMRD::AcquisitionHeader& header) {
            stats.average.insert(header.idx.average);
            stats.kspace_encode_step_1.insert(header.idx.kspace_encode_step_1);
            stats.kspace_encode_step_2.insert(header.idx.kspace_encode_step_2);
            stats.slice.insert(header.idx.slice);
            stats.contrast.insert(header.idx.contrast);
            stats.phase.insert(header.idx.phase);
            stats.repetition.insert(header.idx.repetition);
            stats.set.insert(header.idx.set);
            stats.segment.insert(header.idx.segment);
        }

        void add_acquisition(AcquisitionBucket& bucket, Core::Acquisition acq) {
            auto& head  = std::get<ISMRMRD::AcquisitionHeader>(acq);
            auto espace = head.encoding_space_ref;

            if (ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION).isSet(head.flags)
                || ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING).isSet(head.flags)) {
                bucket.ref_.push_back(acq);
                if (bucket.refstats_.size() < (espace + 1)) {
                    bucket.refstats_.resize(espace + 1);
                }
                add_stats(bucket.refstats_[espace], head);
            }
            if (!(ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION).isSet(head.flags)
                    || ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA).isSet(head.flags))) {
                if (bucket.datastats_.size() < (espace + 1)) {
                    bucket.datastats_.resize(espace + 1);
                }
                add_stats(bucket.datastats_[espace], head);
                bucket.data_.emplace_back(std::move(acq));
            }
        }




    }

    void AccumulateFlagTriggerGadget::send_data(Core::OutputChannel& out, std::map<unsigned short, AcquisitionBucket>& buckets,
                                                       std::vector<Core::Waveform>& waveforms) {
        trigger_events++;
        GDEBUG("Trigger (%d) occurred, sending out %d buckets\n", trigger_events, buckets.size());
        buckets.begin()->second.waveform_ = std::move(waveforms);
        // Pass all buckets down the chain
        for (auto& bucket : buckets)
            out.push(std::move(bucket.second));

        buckets.clear();
    }
    void AccumulateFlagTriggerGadget ::process(
        Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>>& in, Core::OutputChannel& out) {

        auto waveforms = std::vector<Core::Waveform>{};
        auto trigger   = get_trigger(*this);
        auto bucket = AcquisitionBucket{};

        for (auto message : in) {
            if (Core::holds_alternative<Core::Waveform>(message)) {
                waveforms.emplace_back(std::move(Core::get<Core::Waveform>(message)));
                continue;
            }

            auto& acq = Core::get<Core::Acquisition>(message);
            if (is_noise(acq))
                continue;

            add_acquisition(bucket, std::move(acq));

            if (trigger_after(trigger, head))
                send_data(out, buckets, waveforms);
        }
        send_data(out,buckets,waveforms);
    }
    GADGETRON_GADGET_EXPORT(AccumulateFlagTriggerGadget);

    namespace {
        const std::map<std::string, TriggerDimension> triggerdimension_from_name = {

            { "kspace_encode_step_1", TriggerDimension::kspace_encode_step_1 },
            { "kspace_encode_step_2", TriggerDimension::kspace_encode_step_2 },
            { "average", TriggerDimension::average }, { "slice", TriggerDimension::slice },
            { "contrast", TriggerDimension::contrast }, { "phase", TriggerDimension::phase },
            { "repetition", TriggerDimension::repetition }, { "set", TriggerDimension::set },
            { "segment", TriggerDimension::segment }, { "user_0", TriggerDimension::user_0 },
            { "user_1", TriggerDimension::user_1 }, { "user_2", TriggerDimension::user_2 },
            { "user_3", TriggerDimension::user_3 }, { "user_4", TriggerDimension::user_4 },
            { "user_5", TriggerDimension::user_5 }, { "user_6", TriggerDimension::user_6 },
            { "user_7", TriggerDimension::user_7 }, { "n_acquisitions", TriggerDimension::n_acquisitions },
            { "none", TriggerDimension::none }, { "", TriggerDimension::none }
        };
    }

    void from_string(const std::string& str, TriggerDimension& trigger) {
        auto lower = str;
        boost::to_lower(lower);
        trigger = triggerdimension_from_name.at(lower);
    }

}
