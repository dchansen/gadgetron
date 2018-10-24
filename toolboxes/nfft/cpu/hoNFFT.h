/**
    \brief CPU implementation of the non-cartesian FFT

    Comparisions were made to the gridkb function provided in the 
    Stanford Medical Image Reconstruction course lecture notes and 
    the Cuda version of the NFFT (cuNFFT)

    Uses the Kaiser Bessel for convolution
*/

#pragma once 

#include "hoNDArray.h"
#include "vector_td.h"
#include "complext.h"
#include <complex>
#include "../NFFT.h"

#include <boost/shared_ptr.hpp>
#include "hoArmadillo.h"

#include "cpunfft_export.h"
#include "hoNFFT_sparseMatrix.h"

namespace Gadgetron{

    /**
        NFFT class declaration
        ----------------------

        REAL: desired precision : float or double only
        D: dimensionality: 1D, 2D, and 3D supported
    */

    template<class REAL, unsigned int D>
    class EXPORTCPUNFFT hoNFFT_plan
    {
        using ComplexType = std::complex<REAL>;

        /**
            Main interface
        */

        public:





            hoNFFT_plan(
                    const vector_td<size_t,D>& matrix_size,
                    const vector_td<size_t,D>& matrix_size_os,
                    REAL W
            );

             hoNFFT_plan(
                    const vector_td<size_t,D>& matrix_size,
                    REAL oversampling_factor,
                    REAL W
            );



            /** 
                Perform NFFT preprocessing for a given trajectory

                \param k: the NFFT non cartesian trajectory
                \param mode: enum specifying the preprocessing mode
            */

            void preprocess(
                const hoNDArray<vector_td<REAL, D>>& k
            );



            /**
                Execute the NFFT

                \param d: the input data array
                \param m: the output matrix
                \param w: optional density compensation if not iterative
                    provide a 0x0 if non density compensation
                \param mode: enum specifyiing the mode of operation
            */

            void compute(
                const hoNDArray<ComplexType> &d,
                hoNDArray<ComplexType> &m,
                const hoNDArray<REAL>* dcw,
                NFFT_comp_mode mode
            );

            void compute(
                const hoNDArray<complext<REAL>> &d,
                hoNDArray<complext<REAL>> &m,
                const hoNDArray<REAL>* dcw,
                NFFT_comp_mode mode
            );

            /**
                To be used by an operator for iterative reconstruction 

                \param in: the input data
                \param out: the data after MH_H has been applied

                Note: dimensions of in and out should be the same
            */
            void mult_MH_M(
                hoNDArray<ComplexType> &in,
                hoNDArray<ComplexType> &out
            );

            void mult_MH_M(
                hoNDArray<complext<REAL>> &in,
                hoNDArray<complext<REAL>> &out
            );

        /**
            Utilities
        */

        public:


            /** 
                Perform standalone convolution

                \param d: input array
                \param m: output array
                \param mode: enum specifying the mode of the convolution
            */

            void convolve(
                const hoNDArray<ComplexType> &d,
                hoNDArray<ComplexType> &m,
                NFFT_conv_mode mode
            );


            /**
                Cartesian fft. Making use of the hoNDFFT class.

                \param d: input and output for he fft 
                \param mode: enum specifying the mode of the fft 
            */

            void fft(
                hoNDArray<ComplexType> &d,
                NFFT_fft_mode mode
            );

            /**
                NFFT deapodization

                \param d: input and output image to be deapodized 
                \param fourierDomain: has data been ffted
            */

            void deapodize(
                hoNDArray<ComplexType> &d,
                bool fourierDomain = false
            );

        /**
            Private implementation methods
        */

        private:



            /**
                Dedicated convolutions

                The two methods below are entirely symmetric in 
                thier implementation. They could probably be
                combined for conciseness.
            */

            void convolve_NFFT_C2NC(
                const hoNDArray<ComplexType> &d,
                hoNDArray<ComplexType> &m
            );

            void convolve_NFFT_NC2C(
                const hoNDArray<ComplexType> &d,
                hoNDArray<ComplexType> &m
            );


            static vector_td<REAL,D> compute_beta(REAL W, const vector_td<size_t,D>& matrix_size, const vector_td<size_t,D>& matrix_size_os);


        /** 
            Implementation variables
        */

        private:

        REAL W;
        vector_td<size_t,D> matrix_size;
        vector_td<size_t,D> matrix_size_os;

        vector_td<REAL,D> beta;
        Gadgetron::NFFT::NFFT_Matrix<REAL> convolution_matrix;
        Gadgetron::NFFT::NFFT_Matrix<REAL> convolution_matrix_T;

        hoNDArray<ComplexType> deapodization_filter_IFFT;
        hoNDArray<ComplexType> deapodization_filter_FFT;
        boost::shared_ptr<hoNDArray<REAL>> density_compensation_weights;

    };

}
