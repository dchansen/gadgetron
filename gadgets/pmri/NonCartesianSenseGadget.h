//
// Created by david on 27/11/2019.
//

#pragma once

#include "PureGadget.h"
#include "mri_core_data.h"
#include "GenericReconJob.h"
#include "cgSolver.h"

namespace Gadgetron {
    template<template<class> class ARRAY, class SENSE>
    class NonCartesianSenseGadget : Core::PureGadget<IsmrmrdImageArray, GenericReconJob> {
    public:
        NODE_PROPERTY(oversampling_factor, float, "Oversampling factor for NFFT", 1.5);
        NODE_PROPERTY(kernel_width, float, "Kernel width for NFFT", 5.5);
        NODE_PROPERTY(save_individual_frames, bool, "Save individual frames", true);
        NODE_PROPERTY(output_convergence, bool, "Ouput convergence information", false);
        NODE_PROPERTY(output_timing, bool, "Output timing information", false);
        NODE_PROPERTY(kappa, float, "Regularization factor kappa", 0.3);
        NODE_PROPERTY(number_of_iterations, int, "Max number of iterations in CG solver", 5);
        NODE_PROPERTY(cg_limit, float, "Residual limit for CG convergence", 1e-6);

        IsmrmrdImageArray process_function(GenericReconJob reconJob) const override {


            auto traj = reconJob.tra_host_;
            auto dcw = reconJob.dcw_host_;
            sqrt_inplace(dcw.get());
            auto csm = reconJob.csm_host_;
            auto device_samples = reconJob.dat_host_;

            cgSolver<ARRAY> solver;
            auto encoding_operator = boost::make_shared<SENSE>();
            auto regularization = boost::make_shared<imageOperator<ARRAY<float>, ARRAY<complext<float>>>();
            regularization->set_weight(kappa);


            solver.set_encoding_operator(encoding_operator);
            solver.add_regularization_operator(regularization);
            solver.set_max_iterations(number_of_iterations);
            solver.set_tc_tolerance(cg_limit);
            solver.set_output_mode(output_convergence ? OUTPUT_VERBOSE : OUTPUT_SILENT);

            auto samples = reconJob.dat_host_->get_size(0);
            auto channels = reconJob.dat_host_->get_size(1);
            auto rotations = samples/reconJob.tra_host_->get_number_of_elements();
            auto frames = reconJob.tra_host_.get_size(1)*rotations;

            auto warp_size = 32;

            auto matrix_size = uint64d2(reconJob.reg_host_->get_size(0),reconJob.reg_host_->get_size(1));
            auto matrix_size_os = uint64d2(
                    ((size_t(std::ceil(matrix_size[0]*oversampling_factor))+warp_size-1)/warp_size)*warp_size,
                    ((size_t(std::ceil(matrix_size[1]*oversampling_factor))+warp_size-1)/warp_size)*warp_size);
            auto image_dims = to_std_vector(matrix_size);
            encoding_operator->set_domain_dimensions(&image_dims);
            encoding_operator->set_codomain_dimensions(reconJob.dat_host_->get_dimensions().get());
            encoding_operator->set_dcw(dcw);
            encoding_operator->set_csm(csm);

            encoding_operator->setup(matrix_size,matrix_size_os,kernel_width);
            encoding_operator->preprocess(traj.get());

            *device_samples *= *dcw;
            boost::shared_ptr<ARRAY<complext<float>> result;
            if (output_timing) {
                GadgetronTimer timer("NonCartesianSense solver");
                auto result = solver.solve(device_samples.get());
            } else {
                result = solver.solve(device_samples.get());
            }




        }

    };
}


