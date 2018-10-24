/** hoNFFT.cpp */

#include "hoNFFT.h"

#include "hoNDFFT.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include "hoNDArray_utils.h"

#include "vector_td_utilities.h"
#include "vector_td_operators.h"
#include "vector_td_io.h"

#include <math.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <boost/make_shared.hpp>

#include "hoNFFT_sparseMatrix.h"
#include <boost/math/constants/constants.hpp>
#include <KaiserBessel_kernel.h>
#include <boost/range/algorithm/transform.hpp>

#include <cpu/hoNDArray_fileio.h>
#include "GadgetronTimer.h"

using namespace std;

namespace Gadgetron {


    namespace {

        template<typename T, unsigned int D>
        struct FFT {
        };

        template<typename T>
        struct FFT<T, 1> {
            using REAL = typename realType<T>::Type;

            static void fft(hoNDArray<T> &array, NFFT_fft_mode mode) {
                if (mode == NFFT_fft_mode::FORWARDS) {
                    hoNDFFT<REAL>::instance()->fft1c(array);
                } else {
                    hoNDFFT<REAL>::instance()->ifft1c(array);
                }
            }
        };

        template<typename T>
        struct FFT<T, 2> {
            using REAL = typename realType<T>::Type;

            static void fft(hoNDArray<T> &array, NFFT_fft_mode mode) {
                if (mode == NFFT_fft_mode::FORWARDS) {
                    hoNDFFT<REAL>::instance()->fft2c(array);
                } else {
                    hoNDFFT<REAL>::instance()->ifft2c(array);
                }
            }
        };

        template<typename T>
        struct FFT<T, 3> {
            using REAL = typename realType<T>::Type;

            static void fft(hoNDArray<T> &array, NFFT_fft_mode mode) {
                if (mode == NFFT_fft_mode::FORWARDS) {
                    hoNDFFT<REAL>::instance()->fft3c(array);
                } else {
                    hoNDFFT<REAL>::instance()->ifft3c(array);
                }
            }
        };

        template<class REAL> hoNDArray<std::complex<REAL>>
        compute_deapodization_filter(const vector_td<size_t,1>& image_dims, const vector_td<REAL,1>& beta, REAL W){

            hoNDArray<std::complex<REAL>> deapodization(to_std_vector(image_dims));
            vector_td<REAL,1> image_dims_real(image_dims);
            for (int x = 0; x < image_dims[0]; x++){
                auto offset = x - image_dims_real[0]/2;
                deapodization(x) = std::abs(offset) < W/2 ? KaiserBessel(offset,image_dims_real[0],REAL(1)/W,beta[0]) : REAL(0);
            }
            return deapodization;
        }
        template<class REAL> hoNDArray<std::complex<REAL>>
        compute_deapodization_filter(const vector_td<size_t,2>& image_dims, const vector_td<REAL,2>& beta, REAL W){

            hoNDArray<std::complex<REAL>> deapodization(to_std_vector(image_dims));
            vector_td<REAL,2> image_dims_real(image_dims);
            for (int y = 0; y < image_dims[1]; y++) {
                auto offset_y = y - image_dims_real[1]/2;
                auto weight_y = std::abs(offset_y) < W/2 ? KaiserBessel(offset_y,image_dims_real[1],REAL(1)/W,beta[1]) : REAL(0);

                for (int x = 0; x < image_dims[0]; x++) {
                    auto offset_x = x - image_dims_real[0]/2;
                    auto weight_x = std::abs(offset_x) < W/2 ? KaiserBessel(offset_x,image_dims_real[0],REAL(1)/W,beta[0]) : REAL(0);

                    deapodization(x,y) = weight_x*weight_y;
                }
            }
            return deapodization;
        }

         template<class REAL> hoNDArray<std::complex<REAL>>
        compute_deapodization_filter(const vector_td<size_t,3>& image_dims, const vector_td<REAL,3>& beta, REAL W){

            hoNDArray<std::complex<REAL>> deapodization(to_std_vector(image_dims));
            vector_td<REAL,3> image_dims_real(image_dims);
            for (int z = 0; z < image_dims[2]; z++) {
                auto offset_z = z - image_dims_real[2]/2;
                auto weight_z = std::abs(offset_z) < W/2 ? KaiserBessel(offset_z,image_dims_real[2],REAL(1)/W,beta[2]) : REAL(0);

                for (int y = 0; y < image_dims[1]; y++) {
                    auto offset_y = y - image_dims_real[1]/2;
                    auto weight_y = std::abs(offset_y) < W/2 ? KaiserBessel(offset_y,image_dims_real[1],REAL(1)/W,beta[1]) : REAL(0);

                    for (int x = 0; x < image_dims[0]; x++) {
                        auto offset_x = x - image_dims_real[0]/2;
                        auto weight_x = std::abs(offset_x) < W/2 ? KaiserBessel(offset_x,image_dims_real[0],REAL(1)/W,beta[0]) : REAL(0);

                        deapodization(x,y,z) = weight_x*weight_y*weight_z;
                    }
                }
            }
            return deapodization;
        }
    }



    template<class REAL, unsigned int D>
    hoNFFT_plan<REAL, D>::hoNFFT_plan(
            const vector_td<size_t, D> &matrix_size,
            const vector_td<size_t, D> &matrix_size_os,
            REAL W
    ) {

        if (W < REAL(1.0))
            throw std::runtime_error("Kernel width must be larger than 1");
        if (matrix_size > matrix_size_os)
            throw std::runtime_error("Oversampled matrix size must be as least as great as the matrix size");


        this->W = W;
        this->matrix_size = matrix_size;
        this->matrix_size_os = matrix_size_os;

        this->beta = compute_beta(W,matrix_size,matrix_size_os);
        this->deapodization_filter_IFFT = compute_deapodization_filter(this->matrix_size_os,this->beta, this->W);
        this->deapodization_filter_FFT = deapodization_filter_IFFT;
        FFT<std::complex<REAL>,D>::fft(deapodization_filter_IFFT,NFFT_fft_mode::BACKWARDS);
        FFT<std::complex<REAL>,D>::fft(deapodization_filter_FFT,NFFT_fft_mode::FORWARDS);

        boost::transform(deapodization_filter_IFFT,deapodization_filter_IFFT.begin(),[](auto val){return REAL(1)/val;});
        boost::transform(deapodization_filter_FFT,deapodization_filter_FFT.begin(),[](auto val){return REAL(1)/val;});
    }

    template<class REAL, unsigned int D>
    hoNFFT_plan<REAL, D>::hoNFFT_plan(const vector_td<size_t, D> &matrix_size, REAL oversampling_factor, REAL W) {

        this->matrix_size = matrix_size;
        this->W = W;
        this->matrix_size_os = vector_td<size_t,D>(vector_td<REAL,D>(matrix_size)*oversampling_factor);

        this->beta = compute_beta(W,matrix_size,matrix_size_os);
         this->deapodization_filter_IFFT = compute_deapodization_filter(this->matrix_size_os,this->beta, this->W);
        this->deapodization_filter_FFT = deapodization_filter_IFFT;

        FFT<std::complex<REAL>,D>::fft(deapodization_filter_IFFT,NFFT_fft_mode::BACKWARDS);
        FFT<std::complex<REAL>,D>::fft(deapodization_filter_FFT,NFFT_fft_mode::FORWARDS);
//        write_nd_array(abs(&deapodization_filter_IFFT).get(),"deapodization.real");

//        write_nd_array(abs(&deapodization_filter_IFFT).get(),"deapodization.real");
        boost::transform(deapodization_filter_IFFT,deapodization_filter_IFFT.begin(),[](auto val){return REAL(1)/val;});
        boost::transform(deapodization_filter_FFT,deapodization_filter_FFT.begin(),[](auto val){return REAL(1)/val;});
    }


    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::preprocess(
            const hoNDArray<vector_td<REAL, D>> &trajectories) {

        GadgetronTimer timer("Preprocess");
        auto trajectories_scaled = trajectories;
        auto matrix_size_os_real = vector_td<REAL,D>(matrix_size_os);
        std::transform(trajectories_scaled.begin(),trajectories_scaled.end(),trajectories_scaled.begin(),[matrix_size_os_real](auto point){
           return (point+REAL(0.5))*matrix_size_os_real;
        });

        convolution_matrix = NFFT::make_NFFT_matrix(trajectories_scaled, this->matrix_size_os, W, beta);
        convolution_matrix_T = NFFT::transpose(convolution_matrix);

    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::compute(
            const hoNDArray<complext<REAL>> &d,
            hoNDArray<complext<REAL>> &m,
            const hoNDArray<REAL> *dcw,
            NFFT_comp_mode mode
    ) {
        const hoNDArray<ComplexType> *pd = reinterpret_cast<const hoNDArray<ComplexType> *>(&d);
        hoNDArray<ComplexType> *pm = reinterpret_cast<hoNDArray<ComplexType> *>(&m);

        this->compute(*pd, *pm, dcw, mode);
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::compute(
            const hoNDArray<ComplexType> &d,
            hoNDArray<ComplexType> &m,
            const hoNDArray<REAL>* dcw,
            NFFT_comp_mode mode
    ) {
        if (d.get_number_of_elements() == 0)
            throw std::runtime_error("Empty data");

        if (m.get_number_of_elements() == 0)
            throw std::runtime_error("Empty gridding matrix");

        hoNDArray<ComplexType> dtmp(d);

        switch (mode) {
            case NFFT_comp_mode::FORWARDS_C2NC: {

                deapodize(dtmp, false);
                fft(dtmp, NFFT_fft_mode::FORWARDS);
                convolve(dtmp, m, NFFT_conv_mode::C2NC);

                if(dcw) m *= *dcw;
                break;
            }
            case NFFT_comp_mode::FORWARDS_NC2C: {

                if (dcw) dtmp *= *dcw;

                convolve(dtmp, m, NFFT_conv_mode::NC2C);
                fft(m, NFFT_fft_mode::FORWARDS);
                deapodize(m, true);

                break;
            }
            case NFFT_comp_mode::BACKWARDS_NC2C: {


                if (dcw) dtmp *= *dcw;
                convolve(dtmp, m, NFFT_conv_mode::NC2C);
                fft(m, NFFT_fft_mode::BACKWARDS);
                deapodize(m,false);

                break;
            }
            case NFFT_comp_mode::BACKWARDS_C2NC: {

                deapodize(dtmp, true);
                fft(dtmp, NFFT_fft_mode::BACKWARDS);
                convolve(d, m, NFFT_conv_mode::C2NC);

                if (dcw) m *= *dcw;
                break;
            }
        };
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::mult_MH_M(
            hoNDArray<complext<REAL>> &in,
            hoNDArray<complext<REAL>> &out
    ) {
        hoNDArray<ComplexType> *pin = reinterpret_cast<hoNDArray<ComplexType> *>(&in);
        hoNDArray<ComplexType> *pout = reinterpret_cast<hoNDArray<ComplexType> *>(&out);

        this->mult_MH_M(*pin, *pout);
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::mult_MH_M(
            hoNDArray<ComplexType> &in,
            hoNDArray<ComplexType> &out
    ) {
        hoNDArray<ComplexType> tmp(to_std_vector(matrix_size_os));
        compute(in, tmp, density_compensation_weights.get(), NFFT_comp_mode::BACKWARDS_NC2C);
        compute(tmp, out,density_compensation_weights.get(), NFFT_comp_mode::FORWARDS_C2NC);
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::convolve(
            const hoNDArray<ComplexType> &d,
            hoNDArray<ComplexType> &m,
            NFFT_conv_mode mode
    ) {
        if (mode == NFFT_conv_mode::NC2C)
            convolve_NFFT_NC2C(d, m);
        else
            convolve_NFFT_C2NC(d, m);
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::fft(
            hoNDArray<ComplexType> &d,
            NFFT_fft_mode mode
    ) {
        GadgetronTimer timer("FFT");
        FFT<std::complex<REAL>, D>::fft(d, mode);
    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::deapodize(
            hoNDArray<ComplexType> &d,
            bool fourierDomain
    ) {
        if (fourierDomain){
            d *= deapodization_filter_FFT;
        } else {
            d *= deapodization_filter_IFFT;
        }
    }



    namespace {
        template<class REAL> void
        matrix_vector_multiply(const Gadgetron::NFFT::NFFT_Matrix<REAL>& matrix, const std::complex<REAL>* vector, std::complex<REAL>* result) {

            for (size_t i = 0; i < matrix.n_cols; i++) {
                auto &row_indices = matrix.indices[i];
                auto &weights = matrix.weights[i];
#pragma omp simd
                for (size_t n = 0; n < row_indices.size(); n++) {
                    result[i] += vector[row_indices[n]] * weights[n];
                }
            }
        }

    }
    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::convolve_NFFT_C2NC(
            const hoNDArray<ComplexType> &cartesian,
            hoNDArray<ComplexType> &non_cartesian
    ) {

        size_t nbatches = cartesian.get_number_of_elements()/convolution_matrix.n_rows;
        assert(nbatches == non_cartesian.get_number_of_elements()/convolution_matrix.n_cols);

        clear(&non_cartesian);
        for (size_t b = 0; b < nbatches; b++) {

            const ComplexType* cartesian_view = cartesian.get_data_ptr()+b*convolution_matrix.n_rows;
            ComplexType* non_cartesian_view = non_cartesian.get_data_ptr()+b*convolution_matrix.n_cols;

            matrix_vector_multiply(convolution_matrix,cartesian_view,non_cartesian_view);
        }

    }

    template<class REAL, unsigned int D>
    void hoNFFT_plan<REAL, D>::convolve_NFFT_NC2C(
            const hoNDArray<ComplexType> &non_cartesian,
            hoNDArray<ComplexType> &cartesian
    ) {
                size_t nbatches = cartesian.get_number_of_elements()/convolution_matrix.n_rows;
        assert(nbatches == non_cartesian.get_number_of_elements()/convolution_matrix.n_cols);
        GadgetronTimer timer("Convolution");
        clear(&cartesian);
#pragma omp parallel for
        for (size_t b = 0; b < nbatches; b++) {

            ComplexType *cartesian_view = cartesian.get_data_ptr() + b * convolution_matrix.n_rows;
            const ComplexType *non_cartesian_view = non_cartesian.get_data_ptr() + b * convolution_matrix.n_cols;

            matrix_vector_multiply(convolution_matrix_T, non_cartesian_view, cartesian_view);

        }
    }


    template<class REAL, unsigned int D>
    vector_td<REAL, D> hoNFFT_plan<REAL, D>::compute_beta(REAL W, const Gadgetron::vector_td<size_t, D> &matrix_size,
                                                          const Gadgetron::vector_td<size_t, D> &matrix_size_os) {
        // Compute Kaiser-Bessel beta paramter according to the formula provided in
        // Beatty et. al. IEEE TMI 2005;24(6):799-808.
        using boost::math::constants::pi;
        vector_td<REAL, D> beta;

        auto alpha = matrix_size_os / matrix_size;

        for (int d = 0; d < D; d++) {
            beta[d] = (pi<REAL>() * std::sqrt(
                    (W * W) / (alpha[d] * alpha[d]) * (alpha[d] - REAL(0.5)) * (alpha[d] - REAL(0.5)) -
                    REAL(0.8)));
        }

        return beta;
    }



}

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<float, 1>;

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<float, 2>;

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<float, 3>;

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<double, 1>;

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<double, 2>;

template
class EXPORTCPUNFFT Gadgetron::hoNFFT_plan<double, 3>;
