#include "gpuSpiralSensePrepGadget.h"
#include "GenericReconJob.h"
#include "cuNDArray_utils.h"
#include "cuNDArray_reductions.h"
#include "vector_td_utilities.h"
#include "hoNDArray_fileio.h"
#include "vector_td.h"
#include "vector_td_operators.h"
#include "check_CUDA.h"
#include "b1_map.h"
#include "GPUTimer.h"
#include "vds.h"
#include "ismrmrd/xml.h"
#include <armadillo>
#include "mri_core_girf_correction.h"

#include <algorithm>
#include <vector>

namespace Gadgetron {

    gpuSpiralSensePrepGadget::gpuSpiralSensePrepGadget()
            : samples_to_skip_start_(0), samples_to_skip_end_(0), samples_per_interleave_(0), prepared_(false),
              use_multiframe_grouping_(false), acceleration_factor_(0) {
    }

    gpuSpiralSensePrepGadget::~gpuSpiralSensePrepGadget() {}

    int gpuSpiralSensePrepGadget::process_config(ACE_Message_Block *mb) {

        int number_of_devices = 0;
        if (cudaGetDeviceCount(&number_of_devices) != cudaSuccess) {
            GDEBUG("Error: unable to query number of CUDA devices.\n");
            return GADGET_FAIL;
        }

        if (number_of_devices == 0) {
            GDEBUG("Error: No available CUDA devices.\n");
            return GADGET_FAIL;
        }

        device_number_ = deviceno.value();

        if (device_number_ >= number_of_devices) {
            GDEBUG("Adjusting device number from %d to %d\n", device_number_, (device_number_ % number_of_devices));
            device_number_ = (device_number_ % number_of_devices);
        }

        if (cudaSetDevice(device_number_) != cudaSuccess) {
            GDEBUG("Error: unable to set CUDA device.\n");
            return GADGET_FAIL;
        }

        cudaDeviceProp deviceProp;
        if (cudaGetDeviceProperties(&deviceProp, device_number_) != cudaSuccess) {
            GDEBUG("Error: unable to query device properties.\n");
            return GADGET_FAIL;
        }

        unsigned int warp_size = deviceProp.warpSize;

        propagate_csm_from_set_ = propagate_csm_from_set.value();

        if (propagate_csm_from_set_ > 0) {
            GDEBUG("Currently, only set 0 can propagate coil sensitivity maps. Set %d was specified.\n",
                   propagate_csm_from_set_);
            return GADGET_FAIL;
        }

        if (propagate_csm_from_set_ >= 0) {
            GDEBUG("Propagating csm from set %d to all sets\n", propagate_csm_from_set_);
        }

        buffer_using_solver_ = buffer_using_solver.value();
        use_multiframe_grouping_ = use_multiframe_grouping.value();

        if (buffer_using_solver_ && !use_multiframe_grouping_) {
            GDEBUG("Enabling 'buffer_using_solver' requires also enabling 'use_multiframe_grouping'.\n");
            return GADGET_FAIL;
        }

        // Start parsing the ISMRMRD XML header
        //

        ISMRMRD::IsmrmrdHeader h;
        ISMRMRD::deserialize(mb->rd_ptr(), h);


        if (h.encoding.size() != 1) {
            GDEBUG("This Gadget only supports one encoding space\n");
            return GADGET_FAIL;
        }

        // Get the encoding space and trajectory description
        ISMRMRD::EncodingSpace e_space = h.encoding[0].encodedSpace;
        ISMRMRD::EncodingSpace r_space = h.encoding[0].reconSpace;
        ISMRMRD::EncodingLimits e_limits = h.encoding[0].encodingLimits;
        ISMRMRD::TrajectoryDescription traj_desc;

        try {
            auto measurement_dependencies = h.measurementInformation.get().measurementDependency;
            auto girf_dependency = std::find_if(measurement_dependencies.begin(),measurement_dependencies.end(),
                    [](auto& meas){ return meas.dependencyType == "GIRF"; });
            if (girf_dependency != measurement_dependencies.end()){
                auto girf_data = GIRF::load_girf(girf_dependency->measurementID);
                if (girf_data) {
                    this->girf_kernel = boost::make_shared<hoNDArray<std::complex<float>>>(girf_data->girf_kernel);
                    this->girf_sampling_time = girf_data->sampling_time_us;
                    GDEBUG("GIRF kernel with measurement ID %s loaded\n",girf_dependency->measurementID.c_str());
                } else {
                    GWARN("GIRF kernel with measurement ID %s not found\n",girf_dependency->measurementID.c_str());
                }
            }
        } catch (...){
            GDEBUG("The measurement information is missing.\n");
        }
        // Determine reconstruction matrix sizes
        //

        kernel_width_ = buffer_convolution_kernel_width.value();
        oversampling_factor_ = buffer_convolution_oversampling_factor.value();

        image_dimensions_recon_.push_back(
                ((static_cast<unsigned int>(std::ceil(e_space.matrixSize.x * reconstruction_os_factor_x.value())) +
                  warp_size - 1) / warp_size) * warp_size);
        image_dimensions_recon_.push_back(
                ((static_cast<unsigned int>(std::ceil(e_space.matrixSize.y * reconstruction_os_factor_y.value())) +
                  warp_size - 1) / warp_size) * warp_size);

        image_dimensions_recon_os_ = uint64d2
                (((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[0] * oversampling_factor_)) + warp_size -
                   1) / warp_size) * warp_size,
                 ((static_cast<unsigned int>(std::ceil(image_dimensions_recon_[1] * oversampling_factor_)) + warp_size -
                   1) / warp_size) * warp_size);

        // In case the warp_size constraint kicked in
        oversampling_factor_ = float(image_dimensions_recon_os_[0]) / float(image_dimensions_recon_[0]);


        if (h.encoding[0].trajectoryDescription) {
            traj_desc = *h.encoding[0].trajectoryDescription;
        } else {
            GDEBUG("Trajectory description missing");
            return GADGET_FAIL;
        }

        if (traj_desc.identifier != "HargreavesVDS2000") {
            GDEBUG("Expected trajectory description identifier 'HargreavesVDS2000', not found.");
            return GADGET_FAIL;
        }


        long interleaves = -1;
        long fov_coefficients = -1;
        long sampling_time_ns = -1;
        double max_grad = -1.0;
        double max_slew = -1.0;
        double fov_coeff = -1.0;
        double kr_max = -1.0;


        for (std::vector<ISMRMRD::UserParameterLong>::iterator i(traj_desc.userParameterLong.begin());
             i != traj_desc.userParameterLong.end(); ++i) {
            if (i->name == "interleaves") {
                interleaves = i->value;
            } else if (i->name == "fov_coefficients") {
                fov_coefficients = i->value;
            } else if (i->name == "SamplingTime_ns") {
                sampling_time_ns = i->value;
            } else {
                GDEBUG("WARNING: unused trajectory parameter %s found\n", i->name.c_str());
            }
        }

        for (std::vector<ISMRMRD::UserParameterDouble>::iterator i(traj_desc.userParameterDouble.begin());
             i != traj_desc.userParameterDouble.end(); ++i) {
            if (i->name == "MaxGradient_G_per_cm") {
                max_grad = i->value;
            } else if (i->name == "MaxSlewRate_G_per_cm_per_s") {
                max_slew = i->value;
            } else if (i->name == "FOVCoeff_1_cm") {
                fov_coeff = i->value;
            } else if (i->name == "krmax_per_cm") {
                kr_max = i->value;
            } else {
                GDEBUG("WARNING: unused trajectory parameter %s found\n", i->name.c_str());
            }
        }

        if ((interleaves < 0) || (fov_coefficients < 0) || (sampling_time_ns < 0) || (max_grad < 0) || (max_slew < 0) ||
            (fov_coeff < 0) || (kr_max < 0)) {
            GDEBUG("Appropriate parameters for calculating spiral trajectory not found in XML configuration\n");
            return GADGET_FAIL;
        }


        Tsamp_ns_ = sampling_time_ns;
        Nints_ = interleaves;
        interleaves_ = static_cast<int>(Nints_);

        gmax_ = max_grad;
        smax_ = max_slew;
        krmax_ = kr_max;
        fov_ = fov_coeff;

        samples_to_skip_start_ = 0; //n.get<int>(std::string("samplestoskipstart.value"))[0];
        samples_to_skip_end_ = -1; //n.get<int>(std::string("samplestoskipend.value"))[0];

        fov_vec_.push_back(r_space.fieldOfView_mm.x);
        fov_vec_.push_back(r_space.fieldOfView_mm.y);
        fov_vec_.push_back(r_space.fieldOfView_mm.z);

        slices_ = e_limits.slice ? e_limits.slice->maximum + 1 : 1;
        sets_ = e_limits.set ? e_limits.set->maximum + 1 : 1;

        buffer_ = std::vector<ACE_Message_Queue<ACE_MT_SYNCH>>(slices_ * sets_);

        image_headers_queue_ = std::vector<ACE_Message_Queue<ACE_MT_SYNCH>>(slices_ * sets_);

        size_t bsize = sizeof(GadgetContainerMessage<ISMRMRD::ImageHeader>) * 100 * Nints_;

        for (unsigned int i = 0; i < slices_ * sets_; i++) {
            image_headers_queue_[i].high_water_mark(bsize);
            image_headers_queue_[i].low_water_mark(bsize);
        }

        GDEBUG("smax:                    %f\n", smax_);
        GDEBUG("gmax:                    %f\n", gmax_);
        GDEBUG("Tsamp_ns:                %d\n", Tsamp_ns_);
        GDEBUG("Nints:                   %d\n", Nints_);
        GDEBUG("fov:                     %f\n", fov_);
        GDEBUG("krmax:                   %f\n", krmax_);
        GDEBUG("samples_to_skip_start_ : %d\n", samples_to_skip_start_);
        GDEBUG("samples_to_skip_end_   : %d\n", samples_to_skip_end_);
        GDEBUG("recon matrix_size_x    : %d\n", image_dimensions_recon_[0]);
        GDEBUG("recon matrix_size_y    : %d\n", image_dimensions_recon_[1]);

        return GADGET_OK;
    }

    int gpuSpiralSensePrepGadget::
    process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
            GadgetContainerMessage<hoNDArray<std::complex<float> > > *m2) {
        // Noise should have been consumed by the noise adjust, but just in case...
        //
        auto &header = *m1->getObjectPtr();
        bool is_noise = header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
        if (is_noise) {
            m1->release();
            return GADGET_OK;
        }


        if (!prepared_) {
            prepare_nfft(header.number_of_samples,header);
        }

        // Allocate host data buffer if it is NULL
        setup_buffers(header);

        // Define some utility variables
        unsigned int samples_to_copy = header.number_of_samples - samples_to_skip_end_;
        unsigned int interleave = header.idx.kspace_encode_step_1;
        unsigned int slice = header.idx.slice;
        unsigned int set = header.idx.set;
        unsigned int samples_per_channel = host_data_buffer_[set * slices_ + slice].get_size(0);

        // Some book-keeping to keep track of the frame count
        interleaves_counter_singleframe_[set * slices_ + slice]++;
        interleaves_counter_multiframe_[set * slices_ + slice]++;

        // Duplicate the profile to avoid double deletion in case problems are encountered below.
        // Enque profile until all profiles for the reconstruction have been received.
        buffer_[set * slices_ + slice].enqueue_tail(duplicate_profile(m1));

        // Copy profile into the accumulation buffer for csm/regularization estimation

        if (samples_to_skip_end_ == -1) {
            samples_to_skip_end_ = header.number_of_samples - samples_per_interleave_;
            GDEBUG("Adjusting samples_to_skip_end_ = %d\n", samples_to_skip_end_);
        }

        {
            auto data_ptr = host_data_buffer_[set * slices_ + slice].get_data_ptr();

            std::complex<float> *profile_ptr = m2->getObjectPtr()->get_data_ptr();
            for (unsigned int c = 0; c < header.active_channels; c++) {
                memcpy(data_ptr + c * samples_per_channel + interleave * samples_to_copy,
                       profile_ptr + c * header.number_of_samples, samples_to_copy * sizeof(std::complex<float>));
            }
        }

        // Have we received sufficient data for a new frame?
        //

        bool is_last_scan_in_slice = header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_SLICE);

        if (is_last_scan_in_slice) {

            // This was the final profile of a frame
            //

            if (Nints_ % interleaves_counter_singleframe_[set * slices_ + slice]) {
                GDEBUG("Unexpected number of interleaves encountered in frame\n");
                return GADGET_FAIL;
            }

            // Has the acceleration factor changed?
            //

            if (acceleration_factor_ != Nints_ / interleaves_counter_singleframe_[set * slices_ + slice]) {
                change_acceleration_factor(header);
            }

            // Prepare an image header for this frame
            //

            auto image_header = make_image_header(header);
            image_headers_queue_[set * slices_ + slice].enqueue_tail(image_header);
            // Check if it is time to reconstruct.
            // I.e. prepare and pass a Sense job downstream...
            //
            if (!use_multiframe_grouping_ ||
                (use_multiframe_grouping_ && interleaves_counter_multiframe_[set * slices_ + slice] == Nints_)) {

                unsigned int num_coils = header.active_channels;

                //

                // Compute coil images from the fully sampled data buffer
                cuNDArray<float_complext> reg_image = make_reg_image(host_data_buffer_[set * slices_ + slice], set,
                                                                     num_coils);
                boost::shared_ptr<hoNDArray<float_complext> > reg_host = reg_image.to_host();
                boost::shared_ptr<hoNDArray<float_complext> > csm_host = csm_->to_host();

                auto queue_data = get_data_from_queues(set, slice, num_coils);
                GadgetContainerMessage<GenericReconJob> *m4 = new GadgetContainerMessage<GenericReconJob>();

                m4->getObjectPtr()->dat_host_ = std::get<0>(queue_data);
                m4->getObjectPtr()->csm_host_ = csm_host;
                m4->getObjectPtr()->reg_host_ = reg_host;
                m4->getObjectPtr()->tra_host_ = std::get<1>(queue_data);
                m4->getObjectPtr()->dcw_host_ = std::get<2>(queue_data);

                // Pull the image headers out of the queue
                //

                long frames_per_reconstruction = (use_multiframe_grouping_) ? acceleration_factor_ : 1;

                if (image_headers_queue_[set * slices_ + slice].message_count() != frames_per_reconstruction) {
                    m4->release();
                    GDEBUG("Unexpected size of image header queue: %d, %d\n",
                           image_headers_queue_[set * slices_ + slice].message_count(), frames_per_reconstruction);
                    return GADGET_FAIL;
                }

                m4->getObjectPtr()->image_headers_ =
                        boost::shared_array<ISMRMRD::ImageHeader>(new ISMRMRD::ImageHeader[frames_per_reconstruction]);

                for (unsigned int i = 0; i < frames_per_reconstruction; i++) {

                    ACE_Message_Block *mbq;

                    if (image_headers_queue_[set * slices_ + slice].dequeue_head(mbq) < 0) {
                        m4->release();
                        GDEBUG("Image header dequeue failed\n");
                        return GADGET_FAIL;
                    }

                    GadgetContainerMessage<ISMRMRD::ImageHeader> *m = AsContainerMessage<ISMRMRD::ImageHeader>(mbq);
                    m4->getObjectPtr()->image_headers_[i] = *m->getObjectPtr();
                    m->release();
                }

                // The Sense Job needs an image header as well.
                // Let us just copy the initial one...

                GadgetContainerMessage<ISMRMRD::ImageHeader> *m3 = new GadgetContainerMessage<ISMRMRD::ImageHeader>;
                *m3->getObjectPtr() = m4->getObjectPtr()->image_headers_[0];
                m3->cont(m4);

                if (this->next()->putq(m3) < 0) {
                    GDEBUG("Failed to put job on queue.\n");
                    m3->release();
                    return GADGET_FAIL;
                }
                interleaves_counter_multiframe_[set * slices_ + slice] = 0;
            }
            interleaves_counter_singleframe_[set * slices_ + slice] = 0;
        }
        m1->release();
        return GADGET_OK;
    }

    void gpuSpiralSensePrepGadget::setup_buffers(const ISMRMRD::AcquisitionHeader &header) {
        if (host_data_buffer_.empty()) {

            std::vector<size_t> data_dimensions = {size_t(samples_per_interleave_ * interleaves_),
                                                   header.active_channels};
            host_data_buffer_ = std::vector<hoNDArray<float_complext>>(slices_ * sets_,
                                                                       hoNDArray<float_complext>(data_dimensions));

            for (auto &buffer : host_data_buffer_)
                std::fill(buffer.begin(), buffer.end(), 0);
        }

        // Allocate various counters if they are NULL
//

        if (image_counter_.empty()) {
            image_counter_ = std::vector<long>(slices_ * sets_, 0);
        }

        if (interleaves_counter_singleframe_.empty()) {
            interleaves_counter_singleframe_ = std::vector<long>(slices_ * sets_, 0);
        }

        if (interleaves_counter_multiframe_.empty()) {
            interleaves_counter_multiframe_ = std::vector<long>(slices_ * sets_, 0);
        }
    }

    cuNDArray<float_complext>
    gpuSpiralSensePrepGadget::make_reg_image(const hoNDArray<float_complext> &buffer, size_t set, size_t num_coils) {
        std::vector<size_t> image_dims{image_dimensions_recon_[0], image_dimensions_recon_[1], num_coils};
        cuNDArray<float_complext> image(image_dims);
        cuNDArray<float_complext> data(buffer);

        nfft_plan_.compute(&data, &image, dcw_buffer_.get(), cuNFFT_plan<float, 2>::NFFT_BACKWARDS_NC2C);

        // Check if we need to compute a new csm
        if (propagate_csm_from_set_ < 0 || propagate_csm_from_set_ == set || !csm_) {
            csm_ = estimate_b1_map<float, 2>(&image); // Estimates csm
        }
        E_->set_csm(csm_);

        // Compute regularization using basic coil combination
//

        image_dims.pop_back();
        cuNDArray<float_complext> reg_image(&image_dims);
        E_->mult_csm_conj_sum(&image, &reg_image);

        if (buffer_using_solver_) {

            // Compute regularization using cg solver
            //

            // Define preconditioning weights
            boost::shared_ptr<cuNDArray<float> > _precon_weights = sum(abs_square(csm_.get()).get(), 2);
            reciprocal_sqrt_inplace(_precon_weights.get());
            boost::shared_ptr<cuNDArray<float_complext> > precon_weights = real_to_complex<float_complext>(
                    _precon_weights.get());
            _precon_weights.reset();
            D_->set_weights(precon_weights);

            // Solve from the plain coil combination
            reg_image = *cg_.solve_from_rhs(&reg_image);
        }
        return reg_image;
    }

    void gpuSpiralSensePrepGadget::change_acceleration_factor(const ISMRMRD::AcquisitionHeader &header) {
        GDEBUG("Change of acceleration factor detected\n");
        acceleration_factor_ = Nints_ / interleaves_counter_singleframe_[header.idx.set * slices_ + header.idx.slice];

        // The encoding operator needs to have its domain/codomain dimensions set accordingly
        if (buffer_using_solver_) {

            std::vector<size_t> domain_dims = image_dimensions_recon_;

            std::vector<size_t> codomain_dims = *host_traj_->get_dimensions();
            codomain_dims.push_back(header.active_channels);

            E_->set_domain_dimensions(&domain_dims);
            E_->set_codomain_dimensions(&codomain_dims);

            cuNDArray<floatd2> traj(*host_traj_);
            E_->preprocess(&traj);
        }
    }

    GadgetContainerMessage<ISMRMRD::ImageHeader> *
    gpuSpiralSensePrepGadget::make_image_header(const ISMRMRD::AcquisitionHeader &acq_header) {
        GadgetContainerMessage<ISMRMRD::ImageHeader> *header_msg = new GadgetContainerMessage<ISMRMRD::ImageHeader>();
        auto &header = *header_msg->getObjectPtr();

        auto set = acq_header.idx.set;
        auto slice = acq_header.idx.slice;

        header.version = acq_header.version;

        header.matrix_size[0] = image_dimensions_recon_[0];
        header.matrix_size[1] = image_dimensions_recon_[1];
        header.matrix_size[2] = acceleration_factor_;

        header.field_of_view[0] = fov_vec_[0];
        header.field_of_view[1] = fov_vec_[1];
        header.field_of_view[2] = fov_vec_[2];

        header.channels = acq_header.active_channels;
        header.slice = acq_header.idx.slice;
        header.set = acq_header.idx.set;

        header.acquisition_time_stamp = acq_header.acquisition_time_stamp;
        memcpy(header.physiology_time_stamp, acq_header.physiology_time_stamp, sizeof(uint32_t) *
                                                                               ISMRMRD::ISMRMRD_PHYS_STAMPS);

        memcpy(header.position, acq_header.position, sizeof(float) * 3);
        memcpy(header.read_dir, acq_header.read_dir, sizeof(float) * 3);
        memcpy(header.phase_dir, acq_header.phase_dir, sizeof(float) * 3);
        memcpy(header.slice_dir, acq_header.slice_dir, sizeof(float) * 3);
        memcpy(header.patient_table_position, acq_header.patient_table_position, sizeof(float) * 3);

        header.data_type = ISMRMRD::ISMRMRD_CXFLOAT;
        header.image_index = image_counter_[set * slices_ + slice]++;
        header.image_series_index = set * slices_ + slice;

        // Enque header until we are ready to assemble a Sense job
        //

        return header_msg;
    }

    void gpuSpiralSensePrepGadget::prepare_nfft(int number_of_samples, const ISMRMRD::AcquisitionHeader& acq_header) {
        int nfov = 1;         /*  number of fov coefficients.             */
        int ngmax = 1e5;       /*  maximum number of gradient samples      */
        double *xgrad;             /*  x-component of gradient.                */
        double *ygrad;             /*  y-component of gradient.                */
        double *x_trajectory;
        double *y_trajectory;
        double *weighting;
        int ngrad;
        //int     count;
        double sample_time = (1.0 * Tsamp_ns_) * 1e-9;

        /*	call c-function here to calculate gradients */
        calc_vds(smax_, gmax_, sample_time, sample_time, Nints_, &fov_, nfov, krmax_, ngmax, &xgrad, &ygrad, &ngrad);
        samples_per_interleave_ = std::min(ngrad, static_cast<int>(number_of_samples));

        GDEBUG("Using %d samples per interleave\n", samples_per_interleave_);

        if (this->girf_kernel) {
            correct_gradients(xgrad, ygrad, ngrad, Tsamp_ns_*1e-3, this->girf_sampling_time, acq_header.read_dir, acq_header.phase_dir,
                              acq_header.slice_dir);
        }

        /* Calcualte the trajectory and weights*/
        calc_traj(xgrad, ygrad, samples_per_interleave_, Nints_, sample_time, krmax_, &x_trajectory, &y_trajectory,
                  &weighting);

        host_traj_ = boost::shared_ptr<hoNDArray<floatd2> >(new hoNDArray<floatd2>);
        host_weights_ = boost::shared_ptr<hoNDArray<float> >(new hoNDArray<float>);

        std::vector<size_t> trajectory_dimensions;
        trajectory_dimensions.push_back(samples_per_interleave_ * Nints_);




        host_traj_->create(&trajectory_dimensions);
        host_weights_->create(&trajectory_dimensions);


        {
            float *co_ptr = reinterpret_cast<float *>(host_traj_->get_data_ptr());
            float *we_ptr = host_weights_->get_data_ptr();

            for (int i = 0; i < (samples_per_interleave_ * Nints_); i++) {
                co_ptr[i * 2] = -x_trajectory[i] / 2;
                co_ptr[i * 2 + 1] = -y_trajectory[i] / 2;
                we_ptr[i] = weighting[i];
            }

            float min_traj = *std::min_element(co_ptr,co_ptr+host_traj_->get_number_of_elements()*2);
            float max_traj = *std::max_element(co_ptr,co_ptr+host_traj_->get_number_of_elements()*2);

            std::transform(co_ptr,co_ptr+host_traj_->get_number_of_elements()*2,co_ptr,
                    [&](auto element){return (element-min_traj)/(max_traj-min_traj)-0.5;});


        }





        delete[] xgrad;
        delete[] ygrad;
        delete[] x_trajectory;
        delete[] y_trajectory;
        delete[] weighting;

        // Setup the NFFT plan
        //

        cuNDArray<floatd2> traj(*host_traj_);
        dcw_buffer_ = boost::shared_ptr<cuNDArray<float> >(new cuNDArray<float>(*host_weights_));

        nfft_plan_.setup(from_std_vector<size_t, 2>(image_dimensions_recon_), image_dimensions_recon_os_,
                         kernel_width_);
        nfft_plan_.preprocess(&traj, cuNFFT_plan<float, 2>::NFFT_PREP_NC2C);

        // Setup the non-Cartesian Sense encoding operator
        //

        E_ = boost::shared_ptr<cuNonCartesianSenseOperator<float, 2> >(new cuNonCartesianSenseOperator<float, 2>);
        E_->setup(from_std_vector<size_t, 2>(image_dimensions_recon_), image_dimensions_recon_os_, kernel_width_);

        // Setup cg solver if the csm/regularization image is to be based hereon
        //

        if (buffer_using_solver_) {

            E_->set_dcw(sqrt(dcw_buffer_.get()));

            D_ = boost::shared_ptr<cuCgPreconditioner<float_complext> >(new cuCgPreconditioner<float_complext>());
            cg_.set_encoding_operator(E_);
            cg_.set_preconditioner(D_);
            cg_.set_max_iterations(2);
            cg_.set_tc_tolerance(1e-6);
            cg_.set_output_mode(decltype(cg_)::OUTPUT_SILENT);
        }

        prepared_ = true;
    }

    GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *
    gpuSpiralSensePrepGadget::duplicate_profile(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *profile) {
        GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *copy =
                new GadgetContainerMessage<ISMRMRD::AcquisitionHeader>();

        GadgetContainerMessage<hoNDArray<std::complex<float> > > *cont_copy =
                new GadgetContainerMessage<hoNDArray<std::complex<float> > >();

        *copy->getObjectPtr() = *profile->getObjectPtr();
        *(cont_copy->getObjectPtr()) = *(AsContainerMessage<hoNDArray<std::complex<float> > >(
                profile->cont())->getObjectPtr());

        copy->cont(cont_copy);
        return copy;
    }

    std::tuple<boost::shared_ptr<hoNDArray<float_complext>>, boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>>
    gpuSpiralSensePrepGadget::get_data_from_queues(size_t set, size_t slice, size_t num_coils) {
        unsigned int profiles_buffered = buffer_[set * slices_ + slice].message_count();


        auto data_host = boost::make_shared<hoNDArray<float_complext>>(
                samples_per_interleave_ * interleaves_counter_singleframe_[set * slices_ + slice] *
                ((use_multiframe_grouping_) ? acceleration_factor_ : 1),
                num_coils
        );

        std::vector<size_t> ddimensions = {(size_t) samples_per_interleave_ *
                                           interleaves_counter_singleframe_[set * slices_ + slice],
                                           (use_multiframe_grouping_) ? (size_t) acceleration_factor_ : 1};

        boost::shared_ptr<hoNDArray<floatd2> > traj_host(new hoNDArray<floatd2>(&ddimensions));
        boost::shared_ptr<hoNDArray<float> > dcw_host(new hoNDArray<float>(&ddimensions));

        for (unsigned int p = 0; p < profiles_buffered; p++) {
            ACE_Message_Block *mbq;
            buffer_[set * slices_ + slice].dequeue_head(mbq);

            GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *acq =
                    AsContainerMessage<ISMRMRD::AcquisitionHeader>(mbq);

            GadgetContainerMessage<hoNDArray<std::complex<float> > > *daq =
                    AsContainerMessage<hoNDArray<std::complex<float> > >(mbq->cont());

            if (!acq || !daq) {
                GDEBUG("Unable to interpret data on message Q\n");
//                        return GADGET_FAIL;
            }

            for (unsigned int c = 0; c < num_coils; c++) {
                float_complext *data_ptr = data_host->get_data_ptr();
                data_ptr += c * samples_per_interleave_ * profiles_buffered + p * samples_per_interleave_;

                std::complex<float> *r_ptr = daq->getObjectPtr()->get_data_ptr();
                r_ptr += c * daq->getObjectPtr()->get_size(0);

                memcpy(data_ptr, r_ptr, samples_per_interleave_ * sizeof(float_complext));
            }

            floatd2 *traj_ptr = traj_host->get_data_ptr();
            traj_ptr += p * samples_per_interleave_;

            floatd2 *t_ptr = host_traj_->get_data_ptr();
            t_ptr += acq->getObjectPtr()->idx.kspace_encode_step_1 * samples_per_interleave_;

            memcpy(traj_ptr, t_ptr, samples_per_interleave_ * sizeof(floatd2));

            float *dcw_ptr = dcw_host->get_data_ptr();
            dcw_ptr += p * samples_per_interleave_;

            float *d_ptr = host_weights_->get_data_ptr();
            d_ptr += acq->getObjectPtr()->idx.kspace_encode_step_1 * samples_per_interleave_;

            memcpy(dcw_ptr, d_ptr, samples_per_interleave_ * sizeof(float));

            mbq->release();
        }
        return std::make_tuple(data_host, traj_host, dcw_host);
    }

    void gpuSpiralSensePrepGadget::correct_gradients(double *x, double *y, size_t num_elements, float grad_samp_ns,
                                                     float girf_samp_ns, const float *read_dir, const float *phase_dir,
                                                     const float *slice_dir) {


        hoNDArray<float> gradients{num_elements, 2};
        for (size_t i = 0; i < num_elements; i++) {
            gradients(i, 0) = x[i];
            gradients(i, 1) = y[i];
        }

        arma::fmat33 rotation_matrix{{read_dir[0],  read_dir[1],  read_dir[2]},
                                     {phase_dir[0], phase_dir[1], phase_dir[2]},
                                     {slice_dir[0], slice_dir[1], slice_dir[2]}
        };

        gradients = GIRF::girf_correct(gradients, *girf_kernel, rotation_matrix, grad_samp_ns, girf_samp_ns, 0);

        for (size_t i = 0; i < num_elements; i++) {
            x[i] = gradients(i, 0);
            y[i] = gradients(i,1);
        }

    }

    GADGET_FACTORY_DECLARE(gpuSpiralSensePrepGadget)
}
