#include "cuNonCartesianSenseOperator.h"
#include "vector_td_utilities.h"

using namespace Gadgetron;


template<class REAL,unsigned int D>
cuNonCartesianSenseOperator<REAL,D>::cuNonCartesianSenseOperator(ConvolutionType conv) : cuSenseOperator<REAL,D>() {

    convolutionType = conv;
    this->is_preprocessed_ = false;
 }


template<class REAL, unsigned int D> void
cuNonCartesianSenseOperator<REAL,D>::setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W )
{  
  this->plan_ = NFFT<cuNDArray,REAL,D>::make_plan( matrix_size, matrix_size_os, W,convolutionType );
}


//
// Instantiations
//

template class cuNonCartesianSenseOperator<float,1>;
template class cuNonCartesianSenseOperator<float,2>;
template class cuNonCartesianSenseOperator<float,3>;
template class cuNonCartesianSenseOperator<float,4>;

template class cuNonCartesianSenseOperator<double,1>;
template class cuNonCartesianSenseOperator<double,2>;
template class cuNonCartesianSenseOperator<double,3>;
template class cuNonCartesianSenseOperator<double,4>;
