/** \file cuNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once

#include "cuNFFT.h"
#include "NonCartesianSenseOperator.h"

namespace Gadgetron{

  template<class REAL, unsigned int D> class cuNonCartesianSenseOperator : public NonCartesianSenseOperator<cuNDArray,REAL,D>
  {
  
  public:
  
    typedef typename uint64d<D>::Type _uint64d;
    typedef typename reald<REAL,D>::Type _reald;

    cuNonCartesianSenseOperator(ConvolutionType conv = ConvolutionType::STANDARD);
    virtual ~cuNonCartesianSenseOperator() = default;
    virtual void setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W ) override;

  protected:

    ConvolutionType convolutionType;
  };
  
}
