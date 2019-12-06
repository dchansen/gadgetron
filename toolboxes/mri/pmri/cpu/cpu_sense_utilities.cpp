//
// Created by dchansen on 12/6/19.
//
#include "cpu_sense_utilities.h"
#include "hoNDArray_iterators.h"
#include <boost/range.hpp>
#include <boost/range/combine.hpp>

namespace Gadgetron { namespace Sense {

    template <class REAL, unsigned int D>
    void csm_mult_M(
        const hoNDArray<complext<REAL>>& in, hoNDArray<complext<REAL>>& out, const hoNDArray<complext<REAL>>& csm) {



        for (auto& out_csm : boost::range::combine(spans(out,in.get_number_of_dimensions()),spans(csm,csm.get_number_of_dimensions()-1))){
            auto& out_view = std::get<0>(out_csm);
            auto& csm_view = std::get<1>(out_csm);
            out_view = in;
            out_view *= csm_view;
        }
    }

    template <class REAL, unsigned int D>
    void csm_mult_MH(const hoNDArray<complext<REAL>>& in, const hoNDArray<complext<REAL>>& out,
        const hoNDArray<complext<REAL>>& csm) {
        clear(&out);

        for (auto& in_csm : boost::range::combine(spans(in,out.get_number_of_dimensions()),spans(csm,csm.get_number_of_dimensions()-1))){
            auto& in_view = std::get<0>(in_csm);
            auto& csm_view = std::get<1>(in_csm);
            auto tmp = in_view;
            multiplyConj(in_view,csm_view,tmp);
            out += tmp;
        }
    }

}}
