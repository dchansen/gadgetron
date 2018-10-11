/** \file cuNFFT.h
    \brief Cuda implementation of the non-Cartesian FFT

    Reference information on the CUDA/GPU implementation of the NFFT can be found in the papers
    
    Accelerating the Non-equispaced Fast Fourier Transform on Commodity Graphics Hardware.
    T.S. Sørensen, T. Schaeffter, K.Ø. Noe, M.S. Hansen. 
    IEEE Transactions on Medical Imaging 2008; 27(4):538-547.
    
    Real-time Reconstruction of Sensitivity Encoded Radial Magnetic Resonance Imaging Using a Graphics Processing Unit.
    T.S. Sørensen, D. Atkinson, T. Schaeffter, M.S. Hansen.
    IEEE Transactions on Medical Imaging 2009; 28(12):1974-1985. 
*/

#pragma once

#include "cuNDArray.h"
#include "vector_td.h"
#include "complext.h"
#include "gpunfft_export.h"

#include <thrust/device_vector.h>
#include <boost/shared_ptr.hpp>
#include "cuSparseMatrix.h"


enum class ConvolutionType {
    STANDARD,
    ATOMIC,
    SPARSE_MATRIX

};

/**
   Enum to specify the preprocessing mode.
*/
enum class NFFT_prep_mode {
    C2NC, /**< preprocess to perform a Cartesian to non-Cartesian NFFT. */
    NC2C, /**< preprocess to perform a non-Cartesian to Cartesian NFFT. */
    ALL /**< preprocess to perform NFFTs in both directions. */
};

enum class NFFT_wipe_mode {
    ALL, /**< delete all internal memory. */
    PREPROCESSING /**< delete internal memory holding the preprocessing data structures. */
};
/**
     Enum defining the desired NFFT operation
  */
enum class NFFT_comp_mode {
    FORWARDS_C2NC, /**< forwards NFFT Cartesian to non-Cartesian. */
    FORWARDS_NC2C, /**< forwards NFFT non-Cartesian to Cartesian. */
    BACKWARDS_C2NC, /**< backwards NFFT Cartesian to non-Cartesian. */
    BACKWARDS_NC2C /**< backwards NFFT non-Cartesian to Cartesian. */
};

/**
      Enum specifying the direction of the NFFT standalone convolution
   */
enum class NFFT_conv_mode {
    C2NC, /**< convolution: Cartesian to non-Cartesian. */
    NC2C /**< convolution: non-Cartesian to Cartesian. */
};
/**
   Enum specifying the direction of the NFFT standalone FFT.
*/
enum class NFFT_fft_mode {
    FORWARDS, /**< forwards FFT. */
    BACKWARDS /**< backwards FFT. */
};





namespace Gadgetron {

namespace cuNFFT {
    template<class REAL, unsigned int D, ConvolutionType CONV>
    struct convolverNC2C;
        template<class REAL, unsigned int D, ConvolutionType CONV>
    struct convolverC2NC;
}


    template<class REAL, unsigned int D>
    class cuNFFT_plan {
    public:


        /**
            Clear internal storage
            \param mode enum class defining the wipe mode
        */
        virtual void wipe(NFFT_wipe_mode mode) = 0;

        /**
            Setup the plan. Please see the constructor taking similar arguments for a parameter description.
        */
        virtual void setup(typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os,
                           REAL W, int device = -1) = 0;


        /**
           Perform NFFT preprocessing for a given trajectory.
           \param trajectory the NFFT non-Cartesian trajectory normalized to the range [-1/2;1/2].
           \param mode enum class specifying the preprocessing mode
        */
        virtual void preprocess(cuNDArray<typename reald<REAL, D>::Type> *trajectory, NFFT_prep_mode mode) = 0;


        /**
           Execute the NFFT.
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights weighing the input samples according to the sampling density.
           If an 0x0-pointer is provided no density compensation is used.
           \param mode enum class specifying the mode of operation.
        */
        virtual void compute(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out,
                             cuNDArray <REAL> *dcw, NFFT_comp_mode mode) = 0;

        /**
           Execute an NFFT iteraion (from Cartesian image space to non-Cartesian Fourier space and back to Cartesian image space).
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights weighing the input samples according to the sampling density.
           If an 0x0-pointer is provided no density compensation is used.
           \param[in] halfway_dims specifies the dimensions of the intermediate Fourier space (codomain).
        */
        virtual void mult_MH_M(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out,
                               cuNDArray <REAL> *dcw, std::vector<size_t> halfway_dims) = 0;

    public: // Utilities


        /**
           Perform "standalone" convolution
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights.
           \param[in] mode enum class specifying the mode of the convolution
           \param[in] accumulate specifies whether the result is added to the output (accumulation) or if the output is overwritten.
        */
        virtual void convolve(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out, cuNDArray <REAL> *dcw,
                              NFFT_conv_mode mode, bool accumulate = false) = 0;


        /**
           Cartesian FFT. For completeness, just invokes the cuNDFFT class.
           \param[in,out] data the data for the inplace FFT.
           \param mode enum class specifying the direction of the FFT.
           \param do_scale boolean specifying whether FFT normalization is desired.
        */
        virtual void fft(cuNDArray <complext<REAL>> *data, NFFT_fft_mode mode, bool do_scale = true) = 0;

        /**
           NFFT deapodization.
           \param[in,out] image the image to be deapodized (inplace).
        */
        virtual void deapodize(cuNDArray <complext<REAL>> *image, bool fourier_domain = false) = 0;


    public: // Setup queries

        /**
           Get the matrix size.
        */
        inline typename uint64d<D>::Type get_matrix_size() {
            return matrix_size;
        }

        /**
           Get the oversampled matrix size.
        */
        inline typename uint64d<D>::Type get_matrix_size_os() {
            return matrix_size_os;
        }

        /**
           Get the convolution kernel size
        */
        inline REAL get_W() {
            return W;
        }

        /**
           Get the assigned device id
        */
        inline unsigned int get_device() {
            return device;
        }

        /**
           Query of the plan has been setup
        */
        inline bool is_setup() {
            return initialized;
        }

    protected:

        typename uint64d<D>::Type matrix_size;          // Matrix size
        typename uint64d<D>::Type matrix_size_os;       // Oversampled matrix size
        int device;
        bool initialized;
        REAL W;


    };

    /** \class cuNFFT_impl
        \brief Cuda implementation of the non-Cartesian FFT

        ------------------------------
        --- NFFT class declaration ---
        ------------------------------
        REAL:  desired precision : float or double
        D:  dimensionality : { 1,2,3,4 }
        ATOMICS: use atomic device memory transactions : { true, false }

        For the tested hardware the implementation using atomic operations is slower as its non-atomic counterpart.
        However, using atomic operations has the advantage of not requiring any pre-processing.
        As the preprocessing step can be quite costly in terms of memory usage,
        the atomic mode can be necessary for very large images or for 3D/4D volumes.
        Notice: currently no devices support atomics operations in double precision.
    */
    template<class REAL, unsigned int D, ConvolutionType CONV = ConvolutionType::STANDARD>
    class EXPORTGPUNFFT cuNFFT_impl : public cuNFFT_plan<REAL, D> {

    public: // Main interface

        /**
            Default constructor
        */
        cuNFFT_impl();

        /**
           Constructor defining the required NFFT parameters.
           \param matrix_size the matrix size to use for the NFFT. Define as a multiple of 32.
           \param matrix_size_os intermediate oversampled matrix size. Define as a multiple of 32.
           The ratio between matrix_size_os and matrix_size define the oversampling ratio for the NFFT implementation.
           Use an oversampling ratio between 1 and 2. The higher ratio the better quality results,
           however at the cost of increased execution times.
           \param W the concolution window size used in the NFFT implementation.
           The larger W the better quality at the cost of increased runtime.
           \param device the device (GPU id) to use for the NFFT computation.
           The default value of -1 indicates that the currently active device is used.
        */
        cuNFFT_impl(typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os,
                    REAL W, int device = -1);

        /**
           Destructor
        */
        virtual ~cuNFFT_impl();

        /**
            Clear internal storage
            \param mode enum class defining the wipe mode
        */
        virtual void wipe(NFFT_wipe_mode mode) override;

        /**
            Setup the plan. Please see the constructor taking similar arguments for a parameter description.
        */
        virtual void setup(typename uint64d<D>::Type matrix_size, typename uint64d<D>::Type matrix_size_os,
                           REAL W, int device = -1) override;


        /**
           Perform NFFT preprocessing for a given trajectory.
           \param trajectory the NFFT non-Cartesian trajectory normalized to the range [-1/2;1/2].
           \param mode enum class specifying the preprocessing mode
        */
        virtual void preprocess(cuNDArray<typename reald<REAL, D>::Type> *trajectory, NFFT_prep_mode mode) override;


        /**
           Execute the NFFT.
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights weighing the input samples according to the sampling density.
           If an 0x0-pointer is provided no density compensation is used.
           \param mode enum class specifying the mode of operation.
        */
        virtual void compute(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out,
                             cuNDArray <REAL> *dcw, NFFT_comp_mode mode) override;

        /**
           Execute an NFFT iteraion (from Cartesian image space to non-Cartesian Fourier space and back to Cartesian image space).
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights weighing the input samples according to the sampling density.
           If an 0x0-pointer is provided no density compensation is used.
           \param[in] halfway_dims specifies the dimensions of the intermediate Fourier space (codomain).
        */
        virtual void mult_MH_M(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out,
                               cuNDArray <REAL> *dcw, std::vector<size_t> halfway_dims) override;

    public: // Utilities



        /**
           Perform "standalone" convolution
           \param[in] in the input array.
           \param[out] out the output array.
           \param[in] dcw optional density compensation weights.
           \param[in] mode enum class specifying the mode of the convolution
           \param[in] accumulate specifies whether the result is added to the output (accumulation) or if the output is overwritten.
        */
        virtual void convolve(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out, cuNDArray <REAL> *dcw,
                              NFFT_conv_mode mode, bool accumulate = false) override;


        /**
           Cartesian FFT. For completeness, just invokes the cuNDFFT class.
           \param[in,out] data the data for the inplace FFT.
           \param mode enum class specifying the direction of the FFT.
           \param do_scale boolean specifying whether FFT normalization is desired.
        */
        virtual void fft(cuNDArray <complext<REAL>> *data, NFFT_fft_mode mode, bool do_scale = true) override;

        /**
           NFFT deapodization.
           \param[in,out] image the image to be deapodized (inplace).
        */
        virtual void deapodize(cuNDArray <complext<REAL>> *image, bool fourier_domain = false);


        friend class cuNFFT::convolverNC2C<REAL, D, CONV>;
        friend class cuNFFT::convolverC2NC<REAL, D, CONV>;

    private: // Internal to the implementation

        void check_consistency(cuNDArray <complext<REAL>> *samples, cuNDArray <complext<REAL>> *image,
                               cuNDArray <REAL> *dcw);

        // Shared barebones constructor
        void barebones();

        // Compute beta control parameter for Kaiser-Bessel kernel
        void compute_beta();

        // Compute deapodization filter
        boost::shared_ptr<cuNDArray < complext < REAL> > >

        compute_deapodization_filter(bool FFTed = false);

        // Dedicated computes
        void compute_NFFT_C2NC(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out);

        void compute_NFFT_NC2C(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out);

        void compute_NFFTH_NC2C(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out);

        void compute_NFFTH_C2NC(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out);


        // Internal utility
        void image_wrap(cuNDArray <complext<REAL>> *in, cuNDArray <complext<REAL>> *out, bool accumulate);

    private:


        typename uint64d<D>::Type matrix_size_wrap;     // Wrap size at border

        typename reald<REAL, D>::Type alpha;           // Oversampling factor (for each dimension)
        typename reald<REAL, D>::Type beta;            // Kaiser-Bessel convolution kernel control parameter


        unsigned int number_of_samples;               // Number of samples per frame per coil
        unsigned int number_of_frames;                // Number of frames per reconstruction

        cuNFFT::convolverNC2C<REAL,D,CONV> convNC2C;
        cuNFFT::convolverC2NC<REAL,D,CONV> convC2NC;
        //
        // Internal data structures for convolution and deapodization
        //

        boost::shared_ptr<cuNDArray < complext < REAL> > >
        deapodization_filter; //Inverse fourier transformed deapodization filter

        boost::shared_ptr<cuNDArray < complext < REAL> > >
        deapodization_filterFFT; //Fourier transformed deapodization filter

        thrust::device_vector<vector_td<REAL, D>> trajectory_positions;


        //
        // State variables
        //

        bool preprocessed_C2NC, preprocessed_NC2C;

    };

    // Pure virtual class to cause compile errors if you try to use NFFT with double and atomics
    // - since this is not supported on the device
    template<unsigned int D>
    class EXPORTGPUNFFT cuNFFT_impl<double, D, ConvolutionType::ATOMIC> {
        virtual void atomics_not_supported_for_type_double() = 0;
    };

    template<class REAL, unsigned int D> EXPORTGPUNFFT
    boost::shared_ptr<cuNFFT_plan<REAL,D>> make_cuNFFT_plan(ConvolutionType conv = ConvolutionType::STANDARD);


    namespace cuNFFT {

        template<class REAL, unsigned int D, ConvolutionType>
        class convolverNC2C {
        };


        template<class REAL, unsigned int D>
        class convolverNC2C<REAL, D, ConvolutionType::STANDARD> {
        public:
            void
            convolve_NC2C(cuNFFT_impl<REAL, D, ConvolutionType::STANDARD> *plan, cuNDArray<complext<REAL>> *samples,
                          cuNDArray<complext<REAL>> *image, bool accumulate);

            void prepare(cuNFFT_impl<REAL,D,ConvolutionType::STANDARD>* plan, const thrust::device_vector<vector_td<REAL,D>> &trajectory);

        protected:
            thrust::device_vector<unsigned int> tuples_last;
            thrust::device_vector<unsigned int> bucket_begin, bucket_end;


        };



        template<unsigned int D>
        class convolverNC2C<float, D, ConvolutionType::ATOMIC> {
        public:
            void
            convolve_NC2C(cuNFFT_impl<float, D, ConvolutionType::ATOMIC> *plan, cuNDArray<complext<float>> *samples,
                          cuNDArray<complext<float>> *image, bool accumulate);

            void prepare(cuNFFT_impl<float,D,ConvolutionType::ATOMIC>* plan, const thrust::device_vector<vector_td<float,D>> &trajectory){};
        };

        template<class REAL, unsigned int D>
        class convolverNC2C<REAL, D, ConvolutionType::SPARSE_MATRIX> {
        public:
            void convolve_NC2C(cuNFFT_impl<REAL, D, ConvolutionType::SPARSE_MATRIX> *plan,
                               cuNDArray<complext<REAL>> *samples,
                               cuNDArray<complext<REAL>> *image, bool accumulate);

            void prepare(cuNFFT_impl<REAL,D,ConvolutionType::SPARSE_MATRIX>* plan, const thrust::device_vector<vector_td<REAL,D>> &trajectory);

        protected:
            cuCsrMatrix<complext<REAL>> matrix;
            cuCsrMatrix<complext<REAL>> transposed;
        };

        template<class REAL, unsigned int D, ConvolutionType CONV>
        class convolverC2NC {
        public:
            void convolve_C2NC(cuNFFT_impl<REAL,D,CONV>* plan,cuNDArray<complext<REAL>> * image, cuNDArray<complext<REAL>> *samples, bool accumulate);
        };


    }
}
