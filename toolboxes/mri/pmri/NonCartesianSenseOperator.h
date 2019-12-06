//
// Created by dchansen on 12/6/19.
//

#pragma once

#pragma once


namespace Gadgetron{

    template<class ARRAY,unsigned int D> class NonCartesianSenseOperator : public senseOperator<ARRAY,D>
{

    public:
    using REAL = realType_t<ARRAY::element_type>;
    typedef typename uint64d<D>::Type _uint64d;
    typedef typename reald<REAL,D>::Type _reald;

    cuNonCartesianSenseOperator(ConvolutionType conv = ConvolutionType::STANDARD);
    virtual ~cuNonCartesianSenseOperator() {}

    inline boost::shared_ptr< NFFT_plan<REAL, D> > get_plan() { return plan_; }
    inline boost::shared_ptr< cuNDArray<REAL> > get_dcw() { return dcw_; }
    inline bool is_preprocessed() { return is_preprocessed_; }

    virtual void mult_M( cuNDArray< complext<REAL> >* in, cuNDArray< complext<REAL> >* out, bool accumulate = false );
    virtual void mult_MH( cuNDArray< complext<REAL> >* in, cuNDArray< complext<REAL> >* out, bool accumulate = false );

    virtual void setup( _uint64d matrix_size, _uint64d matrix_size_os, REAL W );
    virtual void preprocess( cuNDArray<_reald> *trajectory );
    virtual void set_dcw( boost::shared_ptr< cuNDArray<REAL> > dcw );



    protected:
    boost::shared_ptr< cuNFFT_plan<REAL, D> > plan_;
    boost::shared_ptr< cuNDArray<REAL> > dcw_;
    ConvolutionType convolutionType;
    bool is_preprocessed_;
};
