//
// Created by dchansen on 10/2/18.
//

#include "TrajectoryParameters.h"

namespace Gadgetron {
    namespace Spiral {

        std::pair<hoNDArray<floatd2>, hoNDArray<float>>
        TrajectoryParameters::calculate_trajectories_and_weight(const ISMRMRD::AcquisitionHeader &acq_header) {
            int nfov = 1;         /*  number of fov coefficients.             */
            int ngmax = 1e5;       /*  maximum number of gradient samples      */
            double *xgrad;             /*  x-component of gradient.                */
            double *ygrad;             /*  y-component of gradient.                */
            int ngrad;
            //int     count;
            double sample_time = (1.0 * Tsamp_ns_) * 1e-9;

            /*	call c-function here to calculate gradients */
            calc_vds(smax_, gmax_, sample_time, sample_time, Nints_, &fov_, nfov, krmax_, ngmax, &xgrad, &ygrad,
                     &ngrad);
            int samples_per_interleave_ = std::min(ngrad, static_cast<int>(acq_header.number_of_samples));

            GDEBUG("Using %d samples per interleave\n", samples_per_interleave_);

            auto gradients = create_rotations(xgrad, ygrad, samples_per_interleave_, Nints_);

            delete[] xgrad;
            delete[] ygrad;
//
            if (this->girf_kernel) {
                gradients = correct_gradients(gradients, Tsamp_ns_ * 1e-3, this->girf_sampling_time_us,
                                              acq_header.read_dir, acq_header.phase_dir,
                                              acq_header.slice_dir);
            }


            auto trajectories = calculate_trajectories(gradients, sample_time, krmax_);
            auto weights = calculate_weights(gradients, trajectories);

            {
                float *co_ptr = reinterpret_cast<float *>(trajectories.get_data_ptr());
                float min_traj = *std::min_element(co_ptr, co_ptr + trajectories.get_number_of_elements() * 2);
                float max_traj = *std::max_element(co_ptr, co_ptr + trajectories.get_number_of_elements() * 2);

                std::transform(co_ptr, co_ptr + trajectories.get_number_of_elements() * 2, co_ptr,
                               [&](auto element) { return (element - min_traj) / (max_traj - min_traj) - 0.5; });
            }


            return std::make_pair(std::move(trajectories), std::move(weights));


        }


        TrajectoryParameters::TrajectoryParameters(const ISMRMRD::IsmrmrdHeader &h) {
            ISMRMRD::TrajectoryDescription traj_desc;

            if (h.encoding[0].trajectoryDescription) {
                traj_desc = *h.encoding[0].trajectoryDescription;
            } else {
                throw std::runtime_error("Trajectory description missing");
            }

            if (traj_desc.identifier != "HargreavesVDS2000") {
                throw std::runtime_error("Expected trajectory description identifier 'HargreavesVDS2000', not found.");
            }


            try {
                auto userparam_long = to_map(traj_desc.userParameterLong);
                auto userparam_double = to_map(traj_desc.userParameterDouble);
                Tsamp_ns_ = userparam_long.at("SamplingTime_ns");
                Nints_ = userparam_long.at("interleaves");

                gmax_ = userparam_double.at("MaxGradient_G_per_cm");
                smax_ = userparam_double.at("MaxSlewRate_G_per_cm_per_s");
                krmax_ = userparam_double.at("krmax_per_cm");
                fov_ = userparam_double.at("FOVCoeff_1_cm");
            } catch (std::out_of_range exception) {
                std::string s = "Missing user parameters: " + std::string(exception.what());
                throw std::runtime_error(s);

            }


            TE_ = h.sequenceParameters->TE->at(0);


            if (h.userParameters) {
                try {
                    auto user_params_string = to_map(h.userParameters->userParameterString);
                    auto user_params_double = to_map(h.userParameters->userParameterDouble);

                    auto girf_kernel_string = user_params_string.at("GIRF_kernel");
                    this->girf_kernel = boost::make_optional<hoNDArray<std::complex<float>>>(
                            GIRF::load_girf_kernel(girf_kernel_string));

                    girf_sampling_time_us = user_params_double.at("GIRF_sampling_time_us");

                } catch (std::out_of_range exception) { }
            }

            GDEBUG("smax:                    %f\n", smax_);
            GDEBUG("gmax:                    %f\n", gmax_);
            GDEBUG("Tsamp_ns:                %d\n", Tsamp_ns_);
            GDEBUG("Nints:                   %d\n", Nints_);
            GDEBUG("fov:                     %f\n", fov_);
            GDEBUG("krmax:                   %f\n", krmax_);
            GDEBUG("GIRF kernel:             %d\n", bool(this->girf_kernel));

        }

        hoNDArray<floatd2>
        TrajectoryParameters::correct_gradients(const hoNDArray<floatd2> &gradients, float grad_samp_us,
                                                float girf_samp_us, const float *read_dir, const float *phase_dir,
                                                const float *slice_dir) {

            arma::fmat33 rotation_matrix{{read_dir[0],  read_dir[1],  read_dir[2]},
                                         {phase_dir[0], phase_dir[1], phase_dir[2]},
                                         {slice_dir[0], slice_dir[1], slice_dir[2]}
            };

            return GIRF::girf_correct(gradients, *girf_kernel, rotation_matrix, grad_samp_us, girf_samp_us, 0.85);

        }
    }
}