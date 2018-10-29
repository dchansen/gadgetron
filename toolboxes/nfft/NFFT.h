#pragma once
 /**
   Enum specifying the direction of the NFFT standalone FFT.
*/
namespace Gadgetron {
    enum class NFFT_fft_mode {
        FORWARDS, /**< forwards FFT. */
        BACKWARDS /**< backwards FFT. */
    };

    /**
      Enum specifying the direction of the NFFT standalone convolution
   */
    enum class NFFT_conv_mode {
        C2NC, /**< convolution: Cartesian to non-Cartesian. */
        NC2C /**< convolution: non-Cartesian to Cartesian. */
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
   Enum to specify the preprocessing mode.
*/
    enum class NFFT_prep_mode {
        C2NC, /**< preprocess to perform a Cartesian to non-Cartesian NFFT. */
        NC2C, /**< preprocess to perform a non-Cartesian to Cartesian NFFT. */
        ALL /**< preprocess to perform NFFTs in both directions. */
    };


    template<template<class> class ARRAY,class REAL, unsigned int D>
    struct NFFT {

    };


}
