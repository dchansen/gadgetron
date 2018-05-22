//
// Created by dchansen on 5/18/18.
//
#include "AlphaBetaSwap.h"

using namespace Gadgetron;


static void update_regularization_edge(ImageGraph& graph, const hoNDArray<uint16_t >& fm_index, const hoNDArray<float>& second_deriv,const std::vector<float>& field_map_strength, size_t idx, size_t idx2,size_t edge_idx, uint16_t alpha, uint16_t beta, float scaling){


    auto f_value_alpha = field_map_strength[alpha];
    auto f_value_beta = field_map_strength[beta];

    auto& capacity_map = graph.edge_capacity_map;


    float lambda = std::max(std::min(second_deriv[idx], second_deriv[idx2]), 0.0f);

    if (fm_index[idx2] == alpha || fm_index[idx2] == beta ) {

        auto weight = lambda * scaling * std::norm(f_value_alpha - f_value_beta);
        capacity_map[edge_idx] += weight;
    } else {

        capacity_map[graph.edge_from_sink(idx)] += lambda*scaling*std::norm(f_value_alpha-fm_index[idx2]);
        capacity_map[graph.edge_to_source(idx)] += lambda*scaling*std::norm(f_value_beta-fm_index[idx2]);
    }

}


static void update_graph(ImageGraph &graph, const hoNDArray <uint16_t> &fm_index, const hoNDArray<float> &residuals,
                         const hoNDArray<float> &second_deriv, const std::vector<float> &field_map_strength,
                         uint16_t alpha, uint16_t beta) {

    const auto dims = *fm_index.get_dimensions();



    const size_t source_idx = fm_index.get_number_of_elements();
    const size_t sink_idx = source_idx+1;



    auto& capacity_map = graph.edge_capacity_map;
    //Add regularization edges
    for (int k2 = 0; k2 < dims[1]; k2++){
        for (int k1 = 0; k1 < dims[0]; k1++){
            if (fm_index(k1,k2) == alpha || fm_index(k1,k2) == beta ) {
                size_t idx = k2 * dims[0] + k1;

                for (int d2 = -1; d2 <= 1; d2++){
                    for (int d1 = -1; d1 <= 1; d1++){
                        if (((k1+d1) >= 0) && ((k1+d1) < dims[0]) && ((k2+d2) >= 0) && ((k2+d2) < dims[1]) && ((d1 !=0 || d2 !=0))){
                            size_t idx2 = idx + d1 + d2*dims[0];
                            size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(d1, d2));
                            update_regularization_edge(graph, fm_index, second_deriv, field_map_strength,idx, idx2, edge,alpha, beta, std::sqrt(float(d1*d1+d2*d2)));
                        }
                    }

                }

                capacity_map[graph.edge_from_source(idx)] += residuals(alpha,k1,k2);

                capacity_map[graph.edge_to_sink(idx)] += residuals(beta,k1,k2);

            }

        }
    }

}


static void clean_graph(ImageGraph& g, uint16_t alpha, uint16_t beta) {

    std::fill(g.edge_capacity_map.begin(),g.edge_capacity_map.end(),0);
}




static std::pair<uint16_t ,uint16_t > alpha_beta_swap(ImageGraph& graph, hoNDArray<uint16_t> &field_map_index,
                                               const hoNDArray<float> &residuals_map, const hoNDArray<float>& second_deriv, std::vector<float> field_map_strengths, uint16_t alpha, uint16_t beta) {




    update_graph(graph,field_map_index, residuals_map, second_deriv, field_map_strengths,alpha,beta);
    auto source = graph.source_vertex;
    auto sink = graph.sink_vertex;

    float flow = boost::boykov_kolmogorov_max_flow(graph,source,sink);

    auto color_map = boost::get(boost::vertex_color,graph);

    // Ok, let's figure out what labels were assigned to the source.


    //And update the field_map
    size_t alpha_voxels = 0;
    size_t beta_voxels = 0;
    for (size_t i = 0; i < field_map_index.get_number_of_elements(); i++){
        auto color = boost::get(color_map,i);
        if (color == boost::default_color_type::black_color) {
            field_map_index[i] = beta;
            beta_voxels++;
        } else if (color == boost::default_color_type::white_color) {
            field_map_index[i] = alpha;
            alpha_voxels++;
        }
    }

    return std::make_pair(alpha_voxels,beta_voxels);

}



hoNDArray<uint16_t > Gadgetron::solve_MRF_alphabeta(unsigned int num_iterations, std::vector<float> field_map_strengths, const hoNDArray<float> &residuals_map, const hoNDArray<float>& second_deriv) {


    hoNDArray<uint16_t> field_map_index(residuals_map.get_size(1),residuals_map.get_size(2));
    field_map_index.fill(field_map_strengths.size()/2);
    std::vector<size_t> histogram(field_map_strengths.size(),0);
    for (auto fi : field_map_index) histogram[fi]++;

    ImageGraph graph = ImageGraph(field_map_index.get_size(0),field_map_index.get_size(1));
    for (int i = 0; i < num_iterations; i++) {


        for (uint16_t alpha = 0; alpha < field_map_strengths.size(); alpha++) {
            for (uint16_t beta = alpha + 1; beta < field_map_strengths.size(); beta++) {

                if (histogram[alpha] || histogram[beta]) {
                    auto alpha_beta = alpha_beta_swap(graph, field_map_index, residuals_map, second_deriv,
                                                      field_map_strengths, alpha, beta);
                    histogram[alpha] = alpha_beta.first;
                    histogram[beta] = alpha_beta.second;
                    clean_graph(graph, alpha, beta);
                }


            }
        }
    }
    return field_map_index;
}
