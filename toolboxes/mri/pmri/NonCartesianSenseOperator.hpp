#pragma once
#include "NonCartesianSenseOperator.h"
#include "vector_td_utilities.h"

namespace Gadgetron {

template<template<class> class ARRAY, class REAL, unsigned int D> void
NonCartesianSenseOperator<ARRAY,REAL,D>::mult_M( ARRAY< complext<REAL> >* in, ARRAY< complext<REAL> >* out, bool accumulate )
{
  if( !in || !out ){
    throw std::runtime_error("NonCartesianSenseOperator::mult_M : 0x0 input/output not accepted");
  }
  if ( !in->dimensions_equal(&this->domain_dims_) || !out->dimensions_equal(&this->codomain_dims_)){
	  throw std::runtime_error("NonCartesianSenseOperator::mult_H: input/output arrays do not match specified domain/codomains");
  }

  std::vector<size_t> full_dimensions = *this->get_domain_dimensions();
  full_dimensions.push_back(this->ncoils_);
  ARRAY< complext<REAL> > tmp(&full_dimensions);
  this->mult_csm( in, &tmp );
  
  // Forwards NFFT

  if( accumulate ){
    ARRAY< complext<REAL> > tmp_out(out->get_dimensions());
    plan_->compute( tmp, tmp_out, dcw_.get(), NFFT_comp_mode::FORWARDS_C2NC );
    *out += tmp_out;
  }
  else
    plan_->compute( tmp, *out, dcw_.get(), NFFT_comp_mode::FORWARDS_C2NC );
}

template<template<class> class ARRAY, class REAL, unsigned int D> void
NonCartesianSenseOperator<ARRAY,REAL,D>::mult_MH( ARRAY< complext<REAL> >* in, ARRAY< complext<REAL> >* out, bool accumulate )
{
  if( !in || !out ){
    throw std::runtime_error("NonCartesianSenseOperator::mult_MH : 0x0 input/output not accepted");
  }

  if ( !in->dimensions_equal(&this->codomain_dims_) || !out->dimensions_equal(&this->domain_dims_)){
	  throw std::runtime_error("NonCartesianSenseOperator::mult_MH: input/output arrays do not match specified domain/codomains");
  }
  std::vector<size_t> tmp_dimensions = *this->get_domain_dimensions();
  tmp_dimensions.push_back(this->ncoils_);
  ARRAY< complext<REAL> > tmp(&tmp_dimensions);

 // Do the NFFT
  plan_->compute( *in, tmp, dcw_.get(), NFFT_comp_mode::BACKWARDS_NC2C );

  if( !accumulate ){
    clear(out);    
  }
  
  this->mult_csm_conj_sum( &tmp, out );
}


template<template<class> class ARRAY, class REAL, unsigned int D> void
NonCartesianSenseOperator<ARRAY,REAL,D>::preprocess( ARRAY<_reald> *trajectory )
{
  if( trajectory == 0x0 ){
    throw std::runtime_error( "NonCartesianSenseOperator: cannot preprocess 0x0 trajectory.");
  }
  
  boost::shared_ptr< std::vector<size_t> > domain_dims = this->get_domain_dimensions();
  if( domain_dims.get() == 0x0 || domain_dims->size() == 0 ){
    throw std::runtime_error("NonCartesianSenseOperator::preprocess : operator domain dimensions not set");
  }
  plan_->preprocess( trajectory, NFFT_prep_mode::ALL );
  is_preprocessed_ = true;
}

template<template<class> class ARRAY, class REAL, unsigned int D> void
NonCartesianSenseOperator<ARRAY,REAL,D>::set_dcw( boost::shared_ptr< ARRAY<REAL> > dcw )
{
  dcw_ = dcw;  
}
}
