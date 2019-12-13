/** \file hoNonCartesianSenseOperator.h
    \brief Non-Cartesian Sense operator, GPU based.
*/

#pragma once

#include "senseOperator.h"
#include "NFFT.h"

namespace Gadgetron{

  template<template<class> class ARRAY, class REAL, unsigned int D> class NonCartesianSenseOperator : public senseOperator<ARRAY<complext<REAL>>,D>
  {
  
  public:

    using _uint64d = vector_td<size_t, D>;
    using _reald = vector_td<REAL,D>;

    virtual ~NonCartesianSenseOperator() = default;
    
    inline boost::shared_ptr< NFFT_plan<ARRAY, REAL, D> > get_plan() { return plan_; }
    inline boost::shared_ptr< ARRAY<REAL> > get_dcw() { return dcw_; }
    inline bool is_preprocessed() { return is_preprocessed_; } 

    void mult_M( ARRAY< complext<REAL> >* in, ARRAY< complext<REAL> >* out, bool accumulate = false ) override ;
    void mult_MH( ARRAY< complext<REAL> >* in, ARRAY< complext<REAL> >* out, bool accumulate = false ) override ;

    virtual void setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W ) = 0;
    void preprocess( ARRAY<_reald> *trajectory );
    void set_dcw( boost::shared_ptr< ARRAY<REAL> > dcw );


  
  protected:

    boost::shared_ptr< NFFT_plan<ARRAY,REAL, D> > plan_;
    boost::shared_ptr< ARRAY<REAL> > dcw_;
    bool is_preprocessed_ = false;
  };
  
}


#include "NonCartesianSenseOperator.hpp"