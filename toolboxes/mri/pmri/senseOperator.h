/** \file senseOperator.h
    \brief Base class for all Sense operators
*/

#pragma once

#include "linearOperator.h"

#include <boost/shared_ptr.hpp>
#include <iostream>

namespace Gadgetron{

  template<class ARRAY_TYPE, unsigned int D> class senseOperator : public linearOperator<ARRAY_TYPE>
  {

  public:
    using REAL = realType_t<typename ARRAY_TYPE::element_type>;
    senseOperator() : linearOperator<ARRAY_TYPE>(), ncoils_(0) {}
    virtual ~senseOperator() = default;

    inline unsigned int get_number_of_coils() { return ncoils_; }
    inline boost::shared_ptr<ARRAY_TYPE> get_csm() { return csm_; }
    
    void set_csm( boost::shared_ptr<ARRAY_TYPE> csm )
    {
      if( csm.get() && csm->get_number_of_dimensions() == D+1 ) {
	csm_ = csm;      
	ncoils_ = csm_->get_size(D);
      }
      else{
	throw std::runtime_error("Error: senseOperator::set_csm : unexpected csm dimensionality");
      }    
    }

    virtual void mult_M( ARRAY_TYPE* in, ARRAY_TYPE* out, bool accumulate = false ) = 0;
    virtual void mult_MH( ARRAY_TYPE* in, ARRAY_TYPE* out, bool accumulate = false ) = 0;

    virtual void mult_csm( ARRAY_TYPE* in, ARRAY_TYPE* out ){
        Sense::csm_mult_M<REAL,D>(*in,*out,*csm_);
    }


    virtual void mult_csm_conj_sum( ARRAY_TYPE* in, ARRAY_TYPE* out){
        Sense::csm_mult_MH<REAL,D>(*in,*out,*csm_);
    };

  protected:

    unsigned int ncoils_;
    boost::shared_ptr< ARRAY_TYPE > csm_;
  };
}
