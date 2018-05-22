//
// Created by dchansen on 5/16/18.
//
#pragma once

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include "hoNDArray.h"
#include "ImageGraph.h"

#include "vector_td.h"


namespace Gadgetron {




    hoNDArray<uint16_t > solve_MRF_alphabeta(unsigned int num_iterations, std::vector<float> field_map_strengths, const hoNDArray<float> &residuals_map, const hoNDArray<float>& second_deriv);






}