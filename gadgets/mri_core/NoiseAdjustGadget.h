#pragma once

#include "GadgetronTimer.h"
#include "Node.h"
#include "Types.h"
#include "gadgetron_mricore_export.h"
#include "hoNDArray.h"

#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include <boost/filesystem/path.hpp>

namespace Gadgetron {

    class NoiseAdjustGadget : public Core::ChannelGadget<Core::Acquisition> {
    public:
        NoiseAdjustGadget(const Core::Context& contex, const Core::GadgetProperties& props);

        void process(Core::InputChannel<Core::Acquisition>& in, Core::OutputChannel& out) override;


    protected:
        NODE_PROPERTY(
            noise_dependency_prefix, std::string, "Prefix of noise depencency file", "GadgetronNoiseCovarianceMatrix");
        NODE_PROPERTY(perform_noise_adjust, bool, "Whether to actually perform the noise adjust", true);
        NODE_PROPERTY(pass_nonconformant_data, bool, "Whether to pass data that does not conform", false);
        NODE_PROPERTY(noise_dwell_time_us_preset, float, "Preset dwell time for noise measurement", 0.0);
        NODE_PROPERTY(scale_only_channels_by_name, std::string, "List of named channels that should only be scaled", "");
        NODE_PROPERTY(noise_dependency_folder, boost::filesystem::path, "Path to the working directory", boost::filesystem::temp_directory_path());


//
//        bool noise_decorrelation_calculated;
//        hoNDArray<std::complex<float>> noise_covariance_matrixf;
//        hoNDArray<std::complex<float>> noise_prewhitener_matrixf;
//        hoNDArray<std::complex<float>> noise_covariance_matrixf_once;
//        std::vector<unsigned int> scale_only_channels;
//
//        unsigned long long number_of_noise_samples;
//        unsigned long long number_of_noise_samples_per_acquisition;
//        float noise_dwell_time_us;
//        float acquisition_dwell_time_us;
//        float noise_bw_scale_factor;
          const float receiver_noise_bandwidth;
          Core::optional<hoNDArray<std::complex<float>>> noise_prewhitener_matrix;
//        bool noiseCovarianceLoaded;
//        bool saved;
//
//        std::string noise_dependency_folder;
//        std::string measurement_id;
//        std::string measurement_id_of_noise_dependency;
//        std::string full_name_stored_noise_dependency;



//        bool loadNoiseCovariance();
//        bool saveNoiseCovariance();
//        void computeNoisePrewhitener();

        // We will store/load a copy of the noise scans XML header to enable us to check which coil layout, etc.
        const ISMRMRD::IsmrmrdHeader current_ismrmrd_header;
//        ISMRMRD::IsmrmrdHeader noise_ismrmrd_header;
//        std::vector<size_t> coil_order_of_data_in_noise;


    };
}
