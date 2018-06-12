//
// Created by david on 6/7/2018.
//

#include <random>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include "GraphCutDavid.h"


namespace {
    using namespace Gadgetron;
    static std::mt19937 rng_state(4242);





    template<unsigned int D> void update_regularization_edge(ImageGraph<D> &graph, const hoNDArray<uint16_t> &field_map,
                                    const hoNDArray<uint16_t> &proposed_field_map,
                                    const hoNDArray<float> &second_deriv, const size_t idx, const size_t idx2,
                                    const size_t edge_idx, float scaling) {

        int f_value1 = field_map[idx];
        int pf_value1 = proposed_field_map[idx];
        int f_value2 = field_map[idx2];
        int pf_value2 = proposed_field_map[idx2];
        int a = std::norm(f_value1 - f_value2);
        int b = std::norm(f_value1 - pf_value2);
        int c = std::norm(pf_value1 - f_value2);
        int d = std::norm(pf_value1 - pf_value2);

        float weight = b + c - a - d;

        assert(weight >= 0);
//        weight = std::max(weight,0.0f);
        float lambda = std::max(std::min(second_deriv[idx], second_deriv[idx2]), 0.0f)*scaling;
        weight *= lambda ;

        assert(lambda >= 0);

        auto &capacity_map = graph.edge_capacity_map;

        capacity_map[edge_idx] += weight;
//        capacity_map[graph.reverse(edge_idx)] += weight;

        {
            float aq = lambda * (c - a);

            if (aq > 0) {
                capacity_map[graph.edge_from_source(idx)] += aq;
//            capacity_map[graph.edge_to_source(idx)] += aq;

            } else {
//            capacity_map[graph.edge_from_sink(idx)] -= aq;
                capacity_map[graph.edge_to_sink(idx)] -= aq;
            }
        }

        {
//            float aj = lambda * (std::norm(f_value1 - pf_value2) - std::norm(f_value1 - f_value2));
            float aj = lambda * (d - c);
            if (aj > 0) {
                capacity_map[graph.edge_from_source(idx2)] += aj;
//            capacity_map[graph.edge_to_source(idx2)] += aj;

            } else {
//            capacity_map[graph.edge_from_sink(idx2)] -= aj;
                capacity_map[graph.edge_to_sink(idx2)] -= aj;
            }
        }


    }
    template<unsigned int D> ImageGraph<D> make_graph(const hoNDArray<uint16_t> &field_map, const hoNDArray<uint16_t> &proposed_field_map,
                     const hoNDArray<float> &residual_diff_map, const hoNDArray<float> &second_deriv) {

        const auto dims = from_std_vector<size_t,D>(*field_map.get_dimensions());


        const size_t source_idx = field_map.get_number_of_elements();
        const size_t sink_idx = source_idx + 1;

        ImageGraph<D> graph = ImageGraph<D>(dims);

        auto &capacity_map = graph.edge_capacity_map;
        //Add regularization edges
        for (size_t k2 = 0; k2 < dims[1]; k2++) {
            for (size_t k1 = 0; k1 < dims[0]; k1++) {
                size_t idx = k2 * dims[0] + k1;


                if (k1 < (dims[0] - 1)) {
                    size_t idx2 = idx + 1;
                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(1, 0));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge, 1);
                }


                if (k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0];
                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(0, 1));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge, 1);
                }

//                if (k1 < (dims[0] - 1) && k2 < (dims[1] - 1)) {
//                    size_t idx2 = idx + dims[0] + 1;
//                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(1, 1));
//                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,
//                                               1 / std::sqrt(2.0f));
//                }
//
//                if (k1 < (dims[0] - 1) && k2 > 0) {
//                    size_t idx2 = idx - dims[0] + 1;
//                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(1, -1));
//                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,
//                                               1 / std::sqrt(2.0f));
//                }


                float residual_diff = residual_diff_map[idx];

                if (residual_diff > 0) {
                    capacity_map[graph.edge_from_source(idx)] += int(residual_diff);

                } else {
                    capacity_map[graph.edge_to_sink(idx)] -= int(residual_diff);
                }

            }
        }

        return graph;
    }

}
namespace Gadgetron {
 hoNDArray<uint16_t> update_field_map(const hoNDArray<uint16_t> &field_map_index, const hoNDArray<uint16_t> &proposed_field_map_index,
                                     const hoNDArray<float> &residuals_map, const hoNDArray<float> &second_deriv,
                                     std::vector<float> field_map_strengths) {

//        hoNDArray<float> field_map = create_field_map(field_map_index, field_map_strengths);
//    hoNDArray<float> proposed_field_map = create_field_map(proposed_field_map_index, field_map_strengths);
//        write_nd_array(&proposed_field_map,"proposed.real");
    hoNDArray<float> residual_diff_map(field_map_index.get_dimensions());
    const auto X = field_map_index.get_size(0);
    const auto Y = field_map_index.get_size(1);

    for (size_t k2 = 0; k2 < Y; k2++) {
        for (size_t k1 = 0; k1 < X; k1++) {
            residual_diff_map(k1, k2) = residuals_map(field_map_index(k1, k2), k1, k2) -
                                        residuals_map(proposed_field_map_index(k1, k2), k1, k2);


        }
    }


    ImageGraph<2> graph = make_graph<2>(field_map_index, proposed_field_map_index, residual_diff_map, second_deriv);

    ImageGraph<2>::vertex_descriptor source = graph.source_vertex;
    ImageGraph<2>::vertex_descriptor sink = graph.sink_vertex;

    float flow = boost::boykov_kolmogorov_max_flow(graph, source, sink);

    auto color_map = boost::get(boost::vertex_color, graph);

    // Ok, let's figure out what labels were assigned to the source.
    auto source_label = boost::get(color_map, source);

    auto result = field_map_index;
    //And update the field_map
    size_t updated_voxels = 0;
    for (size_t i = 0; i < field_map_index.get_number_of_elements(); i++) {
        if (boost::get(color_map, i) != boost::default_color_type::black_color) {
            updated_voxels++;
            result[i] = proposed_field_map_index[i];
        }
    }

    return result;

}

}