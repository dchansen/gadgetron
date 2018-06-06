#include "fatwater.h"

#include "hoMatrix.h"
#include "hoNDArray_linalg.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include "hoArmadillo.h"
#include "ImageGraph.h"
#include <boost/config.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
//#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
//#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/timer/timer.hpp>
#include <boost/iterator/function_input_iterator.hpp>
#include <iterator>

// Curve fitting includes (from Hui's example)
//#include "hoNDHarrWavelet.h"
//#include "hoNDRedundantWavelet.h"
#include "hoNDArray_math.h"

#include <boost/random.hpp>
#include <boost/math/constants/constants.hpp>
#include <armadillo>
#include <random>
#include <cpu/hoNDArray_fileio.h>
#include <GadgetronTimer.h>
#include <complex>
#include <cpu/math/hoNDImage_util.h>
#include "FatWaterMixedFitting.h"
#include "ICM.h"
#include "AlphaBetaSwap.h"

//#define GAMMABAR 42.576 // MHz/T


using namespace boost;

//typedef int EdgeWeightType;

//typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
//typedef adjacency_list < vecS, vecS, directedS,
//			 property < vertex_name_t, std::string,
//				    property < vertex_index_t, long,
//			 property < vertex_color_t, boost::default_color_type,
//			 property < vertex_distance_t, long,
//			 property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,
//			 property < edge_capacity_t, EdgeWeightType,
//			 property < edge_residual_capacity_t, EdgeWeightType,
//			 property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

/*
Traits::edge_descriptor AddEdge(Traits::vertex_descriptor &v1,
				Traits::vertex_descriptor &v2,
				property_map < Graph, edge_reverse_t >::type &rev,
				const double capacity,
				Graph &g);
*/
//
//void AddEdge(Traits::vertex_descriptor &v1, Traits::vertex_descriptor &v2, property_map < Graph, edge_reverse_t >::type &rev, const double capacity, Graph &g)
//{
//  Traits::edge_descriptor e1 = add_edge(v1, v2, g).first;
//  Traits::edge_descriptor e2 = add_edge(v2, v1, g).first;
//  put(edge_capacity, g, e1, capacity);
//  put(edge_capacity, g, e2, 0*capacity);
//
//  rev[e1] = e2;
//  rev[e2] = e1;
//}
//
//namespace Gadgetron {

//
//}

namespace Gadgetron {
    using namespace std::complex_literals;
    static constexpr float GAMMABAR = 42.576;
    static constexpr float PI = boost::math::constants::pi<float>();
    static std::mt19937 rng_state(4242);


    void smooth_edges(hoNDArray<float> &residual, float relative_size = 0.1, float relative_strength = 0.5) {

        const int fm = residual.get_size(0);
        const int X = residual.get_size(1);
        const int Y = residual.get_size(2);

        const int edge_sizeX = X * relative_size;
        const int edge_sizeY = Y * relative_size;

//#pragma omp parallel for
        for (int k2 = 0; k2 < Y; k2++) {
            for (int k1 = 0; k1 < X; k1++) {
                if (k1 < edge_sizeX || k1 > (X - edge_sizeX) || k2 < edge_sizeY || k2 > (Y - edge_sizeY)) {
                    for (int k0 = 0; k0 < fm; k0++) {
                        residual(k0, k1, k2) *= relative_strength;
                    }

                }

            }

        }

    }

    void enhance_regularization_edges(hoNDArray<float> &residual, float relative_sigma = 1) {


        const int X = residual.get_size(0);
        const int Y = residual.get_size(1);

        const float sigma_X2 = std::pow(relative_sigma * X / 2, 2);
        const float sigma_Y2 = std::pow(relative_sigma * Y / 2, 2);

#pragma omp parallel for collapse(2)
        for (int k2 = 0; k2 < Y; k2++) {
            for (int k1 = 0; k1 < X; k1++) {
                const float dist =
                        std::pow<float>(k1 - X / 2, 2) / sigma_X2 + std::pow<float>(k2 - Y / 2, 2) / sigma_Y2;
                float weight = std::exp(-0.5 * dist);
                residual(k1, k2) /= weight;
            }
        }

    }

    hoNDArray<uint16_t> create_field_map_proposal1(const hoNDArray<uint16_t> &field_map_index,
                                                   const hoNDArray<std::vector<uint16_t>> &minima,
                                                   const hoNDArray<float> &residuals,
                                                   const std::vector<float> &field_map_strengths, float fat_freq,
                                                   float dF, float dTE) {

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        const size_t field_maps = field_map_strengths.size();
        std::uniform_int_distribution<int> coinflip(0, 1);
        int jump;
        if (coinflip(rng_state)) {
            jump = round(std::abs(fat_freq / dF));
            std::cout << " Jump1 " << jump << "\n";
        } else {
            jump = round((1.0 / dTE - std::abs(fat_freq)) / dF);
            std::cout << " Jump2 " << jump << "\n";
        }


        for (size_t i = 0; i < elements; i++) {
            auto &mins = minima[i];
            auto fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.begin(), mins.end(),
                                     [&](auto fqi) { return fqi > fbi + 20; }); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.end()) ? std::min<int>(fbi + jump, field_maps - 1) : *fqmi;
        }

        return proposed_field_map_index;


    }

    hoNDArray<uint16_t> create_field_map_proposal2(const hoNDArray<uint16_t> &field_map_index,
                                                   const hoNDArray<std::vector<uint16_t>> &minima,
                                                   const hoNDArray<float> &residuals,
                                                   const std::vector<float> &field_map_strengths, float fat_freq,
                                                   float dF, float dTE) {

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        std::uniform_int_distribution<int> coinflip(0, 1);
        int jump;
        if (coinflip(rng_state)) {
            jump = round(std::abs(fat_freq / dF));
        } else {
            jump = round((1.0 / dTE - std::abs(fat_freq)) / dF);
        }

        for (size_t i = 0; i < elements; i++) {
            auto &mins = minima[i];
            int fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.rbegin(), mins.rend(),
                                     [&](auto fqi) { return fqi < (fbi - 20); }); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.rend()) ? std::max<int>(fbi - jump, 0) : *fqmi;
        }

        return proposed_field_map_index;


    }

    hoNDArray<uint16_t>
    create_field_map_proposal_standard(const hoNDArray<uint16_t> &field_map_index, int sign, uint16_t max_field_value) {


        std::uniform_int_distribution<int> rng(1, 3);

        int step_size = sign * rng(rng_state);
//        int step_size = sign;
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        std::transform(field_map_index.begin(), field_map_index.end(), proposed_field_map_index.begin(),
                       [&](uint16_t j) {
                           return uint16_t(std::min(std::max(j + step_size, 0), int(max_field_value)));
                       });

        return proposed_field_map_index;
    }

    hoNDArray<std::vector<uint16_t>> find_local_minima(const hoNDArray<float> &residuals, float threshold = 0.06f) {


        auto threshold_signal = std::move(*sum(&residuals, 0));
        threshold_signal /= max(&threshold_signal);
        sqrt_inplace(&threshold_signal);

        auto min_residuals = std::move(*min(&residuals, 0));
        auto max_residuals = std::move(*max(&residuals, 0));


        const auto Y = residuals.get_size(2);
        const auto X = residuals.get_size(1);
        hoNDArray<std::vector<uint16_t>> result(X, Y);
        const auto steps = residuals.get_size(0);
        for (size_t k2 = 0; k2 < Y; k2++) {
            for (size_t k1 = 0; k1 < X; k1++) {

                std::vector<uint16_t> minima;
                if (threshold_signal(k1, k2) > threshold) {
                    for (size_t k0 = 1; k0 < steps - 1; k0++) {
                        if ((residuals(k0, k1, k2) < residuals(k0 - 1, k1, k2)) &&
                            (residuals(k0 + 1, k1, k2) >= residuals(k0, k1, k2)) &&
                            residuals(k0, k1, k2) <
                            min_residuals(k1, k2) + 0.3 * (max_residuals(k1, k2) - min_residuals(k1, k2))) {
                            minima.push_back(k0);
                        }

                    }
                }
                result(k1, k2) = std::move(minima);
            }
        }
        return result;
    }

    hoNDArray<float> approx_second_derivative(const hoNDArray<float> &residuals,
                                              const hoNDArray<std::vector<uint16_t>> &local_min_indices,
                                              float step_size) {
        hoNDArray<float> second_deriv(local_min_indices.get_dimensions());

        const auto Y = second_deriv.get_size(1);
        const auto X = second_deriv.get_size(0);
        const auto nfields = residuals.get_size(0);

        for (uint16_t k2 = 0; k2 < Y; k2++) {
            for (uint16_t k1 = 0; k1 < X; k1++) {

                auto minimum = std::min_element(&residuals(1, k1, k2), &residuals(nfields - 1, k1, k2)) -
                               &residuals(0, k1, k2);
                /*
                const auto& min_indices = local_min_indices(k1,k2);
                size_t minimum;
                if (min_indices.empty()) {
                    if (residuals(0,k1,k2) < residuals(nfields-1,k1,k2)){
                        minimum = 9;
                    } else {
                        minimum = nfields-10;
                    }

                } else {

                    minimum = *std::min_element(min_indices.begin(), min_indices.end(), [&](auto i, auto j) {
                        return residuals(i, k1, k2) < residuals(j, k1, k2);
                    });
                }
                 */

                auto sd =
                        (residuals(minimum - 1, k1, k2) + residuals(minimum + 1, k1, k2) -
                         2 * residuals(minimum, k1, k2)) / (step_size * step_size);
//                    second_deriv(k1, k2) = std::max(sd,0.0f);
                second_deriv(k1, k2) = sd;

            }
        }

//        second_deriv.fill(1.0f);

        return second_deriv;


    }


    typedef ImageGraph Graph;


    void update_regularization_edge(Graph &graph, const hoNDArray<uint16_t> &field_map,
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
        float lambda = std::max(std::min(second_deriv[idx], second_deriv[idx2]), 0.0f);
        weight *= lambda * scaling;

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


    Graph make_graph(const hoNDArray<uint16_t> &field_map, const hoNDArray<uint16_t> &proposed_field_map,
                     const hoNDArray<float> &residual_diff_map, const hoNDArray<float> &second_deriv) {

        const auto dims = *field_map.get_dimensions();


        const size_t source_idx = field_map.get_number_of_elements();
        const size_t sink_idx = source_idx + 1;

        Graph graph = Graph(dims[0], dims[1]);

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

                if (k1 < (dims[0] - 1) && k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0] + 1;
                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(1, 1));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,
                                               1 / std::sqrt(2.0f));
                }

                if (k1 < (dims[0] - 1) && k2 > 0) {
                    size_t idx2 = idx - dims[0] + 1;
                    size_t edge = graph.edge_from_offset(idx, vector_td<int, 2>(1, -1));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,
                                               1 / std::sqrt(2.0f));
                }


                float residual_diff = residual_diff_map[idx];

                if (residual_diff > 0) {
                    capacity_map[graph.edge_from_source(idx)] += residual_diff;

                } else {
                    capacity_map[graph.edge_to_sink(idx)] -= residual_diff;
                }

            }
        }

        return graph;
    }

    hoNDArray<float>
    create_field_map(const hoNDArray<uint16_t> &field_map_index, const std::vector<float> &field_map_strengths) {
        const uint16_t max_val = field_map_strengths.size() - 1;
        hoNDArray<float> field_map(field_map_index.get_dimensions());
        std::transform(field_map_index.begin(), field_map_index.end(), field_map.begin(),
                       [&](uint16_t i) { return field_map_strengths[std::min(i, max_val)]; });
        return field_map;
    }


    size_t update_field_map(hoNDArray<uint16_t> &field_map_index, const hoNDArray<uint16_t> &proposed_field_map_index,
                            const hoNDArray<float> &residuals_map, const hoNDArray<float> &second_deriv,
                            std::vector<float> field_map_strengths) {

//        hoNDArray<float> field_map = create_field_map(field_map_index, field_map_strengths);
        hoNDArray<float> proposed_field_map = create_field_map(proposed_field_map_index,field_map_strengths);
        write_nd_array(&proposed_field_map,"proposed.real");
        hoNDArray<float> residual_diff_map(field_map_index.get_dimensions());
        const auto X = field_map_index.get_size(0);
        const auto Y = field_map_index.get_size(1);

        for (size_t k2 = 0; k2 < Y; k2++) {
            for (size_t k1 = 0; k1 < X; k1++) {
                residual_diff_map(k1, k2) = residuals_map(field_map_index(k1, k2), k1, k2) -
                                            residuals_map(proposed_field_map_index(k1, k2), k1, k2);


            }
        }


        Graph graph = make_graph(field_map_index, proposed_field_map_index, residual_diff_map, second_deriv);

        Graph::vertex_descriptor source = graph.source_vertex;
        Graph::vertex_descriptor sink = graph.sink_vertex;

        float flow = boost::boykov_kolmogorov_max_flow(graph, source, sink);

        auto color_map = boost::get(vertex_color, graph);

        // Ok, let's figure out what labels were assigned to the source.
        auto source_label = boost::get(color_map, source);

        //And update the field_map
        size_t updated_voxels = 0;
        for (size_t i = 0; i < field_map_index.get_number_of_elements(); i++) {
            if (boost::get(color_map, i) != boost::default_color_type::black_color) {
                updated_voxels++;
                field_map_index[i] = proposed_field_map_index[i];
            }
        }

        return updated_voxels;

    }
//

    arma::Mat<std::complex<float>>
    calculate_psi_matrix(const std::vector<float> &echoTimes,
                         const arma::Mat<std::complex<float>> &phiMatrix, float fm, float r2star) {
        arma::Mat<std::complex<float>> psiMatrix(phiMatrix.n_rows, phiMatrix.n_cols);
        for (int k1 = 0; k1 < phiMatrix.n_rows; k1++) {
            auto curModulation = exp(-r2star * echoTimes[k1] + 2if * PI * echoTimes[k1] * fm);
            for (int k2 = 0; k2 < phiMatrix.n_cols; k2++) {
                psiMatrix(k1, k2) = phiMatrix(k1, k2) * curModulation;
            }
        }


        return psiMatrix;
    }

    hoNDArray<arma::Mat<std::complex<float>>>
    CalculateResidualMap(const std::vector<float> &echoTimes, uint16_t num_r2star, uint16_t num_fm,
                         const arma::Mat<std::complex<float>> &phiMatrix,
                         const std::vector<float> &field_map_strengths, const std::vector<float> &r2stars) {

        hoNDArray<arma::Mat<std::complex<float>>> Ps(num_fm, num_r2star);
        size_t nte = phiMatrix.n_rows;

#pragma omp parallel for collapse(2)
        for (int k3 = 0; k3 < num_fm; k3++) {
            for (int k4 = 0; k4 < num_r2star; k4++) {
                float fm = field_map_strengths[k3];
                float r2star = r2stars[k4];

                arma::Mat<std::complex<float>> psiMatrix = calculate_psi_matrix(echoTimes, phiMatrix, fm,
                                                                                r2star);
                Ps(k3, k4) = arma::eye<arma::Mat<std::complex<float>>>(nte, nte) -
                             psiMatrix * arma::pinv(psiMatrix);

            }
        }
        return Ps;
    }

    hoNDArray<uint16_t>
    solve_MRF(uint16_t num_iterations, const std::vector<float> &field_map_strengths,
              const hoNDArray<float> &residual, const hoNDArray<std::vector<uint16_t>> &local_min_indices,
              const hoNDArray<float> &second_deriv, float fat_freq, float dF, float dTE) {

        hoNDArray<uint16_t> fmIndex(local_min_indices.get_dimensions());

        std::uniform_int_distribution<int> coinflip(0, 2);
        fmIndex.fill(field_map_strengths.size() / 2);

//        fatwaterICM<5>(fmIndex,residual,field_map_strengths,40,10.0);for (int i = 0; i < num_iterations; i++){

        bool up_success = false;
        bool down_success = false;

        for (int i = 0; i < num_iterations; i++) {
            std::cout << "Iteration " << i << std::endl;

            if (coinflip(rng_state) == 0 || i < 15) {
                if (!(i%2)) {
                    std::cout << "Down" << std::endl;
                    auto fmIndex_update = create_field_map_proposal1(fmIndex, local_min_indices, residual,
                                                                     field_map_strengths, fat_freq, dF, dTE);
                    auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv,
                                                    field_map_strengths);
                } else {
                    std::cout << "Up" << std::endl;
                    auto fmIndex_update = create_field_map_proposal2(fmIndex, local_min_indices, residual,
                                                                     field_map_strengths, fat_freq, dF, dTE);
                    auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv,
                                                    field_map_strengths);
                }
            } else {
                auto fmIndex_update = create_field_map_proposal_standard(fmIndex, std::pow(-1, i),
                                                                         field_map_strengths.size() - 1);
                auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv, field_map_strengths);
            }

/*
        if (!down_success){
            std::cout << "Down " << std::endl;
            auto fmIndex_update = create_field_map_proposal1(fmIndex, local_min_indices, residual, field_map_strengths,fat_freq,dF,dTE);
            auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv, field_map_strengths);
            down_success = updated == 0;
        } else if (!up_success) {
            auto fmIndex_update = create_field_map_proposal2(fmIndex, local_min_indices, residual, field_map_strengths,fat_freq,dF,dTE);
            auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv, field_map_strengths);
            up_success = updated == 0;
        } else {
            auto fmIndex_update = create_field_map_proposal_standard(fmIndex, coinflip(rng_state)*2-1, field_map_strengths.size());
            auto updated = update_field_map(fmIndex, fmIndex_update, residual, second_deriv, field_map_strengths);
        }*/

        }

        return fmIndex;
    }


    hoNDArray<float>
    calculate_r2star_map(const hoNDArray<std::complex<float> > &data, const hoNDArray<uint16_t> &fm_index,
                         const std::vector<float> &r2star_values, const std::vector<float> &field_map_strenghts,
                         const arma::Mat<std::complex<float>> &phiMatrix, std::vector<float> &echoTimes) {
        using cMat = arma::Mat<std::complex<float>>;
        uint16_t X = data.get_size(0);
        uint16_t Y = data.get_size(1);
        uint16_t Z = data.get_size(2);
        uint16_t CHA = data.get_size(3);
        uint16_t N = data.get_size(4);
        uint16_t S = data.get_size(5);
        uint16_t LOC = data.get_size(6);
        std::unordered_map<uint16_t, std::vector<arma::Mat<std::complex<float>>>> Ps;
        auto nte = phiMatrix.n_rows;
        for (auto fm : fm_index) {
            if (!Ps.count(fm)) {
                std::vector<arma::Mat<std::complex<float>>> projection_matrices(r2star_values.size());
                std::transform(r2star_values.begin(), r2star_values.end(), projection_matrices.begin(),
                               [&](float r2star) {
                                   auto psiMatrix = calculate_psi_matrix(echoTimes, phiMatrix, field_map_strenghts[fm],
                                                                         r2star);
                                   arma::Mat<std::complex<float>> result =
                                           arma::eye<arma::Mat<std::complex<float>>>(nte, nte) - psiMatrix *
                                                                                                 arma::solve(
                                                                                                         psiMatrix.t() *
                                                                                                         psiMatrix,
                                                                                                         psiMatrix.t());
                                   return result;
                               });
                Ps.emplace(fm, std::move(projection_matrices));
            }
        }

        hoNDArray<float> r2star_map(fm_index.get_dimensions());

#pragma omp parallel for collapse(2)
        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {
                // Get current signal
                std::vector<cMat> signals(CHA, cMat(S, N));
                for (int cha = 0; cha < CHA; cha++) {
                    auto &tempSignal = signals[cha];
                    for (int k4 = 0; k4 < N; k4++) {
                        for (int k5 = 0; k5 < S; k5++) {
                            tempSignal(k5, k4) = data(k1, k2, 0, cha, k4, k5, 0);

                        }
                    }
                }


                float minResidual = std::numeric_limits<float>::max();
                auto &P = Ps[fm_index(k1, k2)];


                for (int kr2 = 0; kr2 < r2star_values.size(); kr2++) {

                    float curResidual = 0;

                    for (int cha = 0; cha < CHA; cha++) {
                        // Apply projector
                        arma::Mat<std::complex<float>> projected = P[kr2] * signals[cha];
                        curResidual += std::accumulate(projected.begin(), projected.end(), 0.0f,
                                                       [](auto v1, auto v2) {
                                                           return v1 +
                                                                  std::norm(v2);
                                                       });
                    }
                    if (curResidual < minResidual) {
                        minResidual = curResidual;
                        r2star_map(k1, k2) = r2star_values[kr2];
                    }

                }


            }
        }

        return r2star_map;
    }


    void CalculateResidualAndR2Star(uint16_t num_r2star, uint16_t num_fm, uint16_t nte,
                                    const hoNDArray<arma::Mat<std::complex<float>>> &Ps,
                                    const hoNDArray<std::complex<float>> &data, hoNDArray<float> &residual,
                                    hoNDArray<uint16_t> &r2starIndex) {

        using cMat = arma::Mat<std::complex<float>>;
        uint16_t X = data.get_size(0);
        uint16_t Y = data.get_size(1);
        uint16_t Z = data.get_size(2);
        uint16_t CHA = data.get_size(3);
        uint16_t N = data.get_size(4);
        uint16_t S = data.get_size(5);
        uint16_t LOC = data.get_size(6);

        r2starIndex = hoNDArray<uint16_t>(data.get_size(0), data.get_size(1), num_fm);
        residual = hoNDArray<float>(num_fm, data.get_size(0), data.get_size(1));


#pragma omp parallel for collapse(2)
        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {
                // Get current signal
//                arma::Mat<std::complex<float>> tempSignal(S, N);
                std::vector<cMat> signals(CHA, cMat(S, N));
                for (int cha = 0; cha < CHA; cha++) {
                    auto &tempSignal = signals[cha];
                    for (int k4 = 0; k4 < N; k4++) {
                        for (int k5 = 0; k5 < S; k5++) {
                            tempSignal(k5, k4) = data(k1, k2, 0, cha, k4, k5, 0);

                        }
                    }
                }

                for (int k3 = 0; k3 < num_fm; k3++) {

                    float minResidual = std::numeric_limits<float>::max();

                    for (int k4 = 0; k4 < num_r2star; k4++) {
                        // Apply projector
                        float curResidual = 0;
                        for (int cha = 0; cha < CHA; cha++) {
                            arma::Mat<std::complex<float>> projected = Ps(k3, k4) * signals[cha];
                            curResidual += std::accumulate(projected.begin(), projected.end(), 0.0f,
                                                           [](auto v1, auto v2) {
                                                               return v1 +
                                                                      std::norm(v2);
                                                           });
                        }
                        if (curResidual < minResidual) {
                            minResidual = curResidual;
                            r2starIndex(k1, k2, k3) = k4;
                        }
                    }
                    residual(k3, k1, k2) = minResidual;

                }
            }
        }
    }

    hoNDArray<std::complex<float>>
    separate_species(const hoNDArray<std::complex<float>> &data, const std::vector<float> &echoTimes,
                     const arma::Mat<std::complex<float>> &phiMatrix, hoNDArray<float> &r2star_map,
                     hoNDArray<float> &field_map) {

        using cMat = arma::Mat<std::complex<float>>;
        uint16_t X = data.get_size(0);
        uint16_t Y = data.get_size(1);
        uint16_t Z = data.get_size(2);
        uint16_t CHA = data.get_size(3);
        uint16_t N = data.get_size(4);
        uint16_t S = data.get_size(5);
        uint16_t LOC = data.get_size(6);
        hoNDArray<std::complex<float> > out(X, Y, Z, CHA, N, 2, LOC); // S dimension gets replaced by water/fat stuff

        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {

                std::vector<cMat> signals(CHA, cMat(S, N));

                // Get current signal
                for (int cha = 0; cha < CHA; cha++) {
                    auto &tempSignal = signals[cha];
                    for (int k4 = 0; k4 < N; k4++) {
                        for (int k5 = 0; k5 < S; k5++) {
                            tempSignal(k5, k4) = data(k1, k2, 0, cha, k4, k5, 0);
                        }
                    }
                }
                // Get current Psi matrix
//                fm = field_map_strengths[fmIndex(k1, k2)];
                auto fm = field_map(k1, k2);
                auto r2star = r2star_map(k1, k2);

                arma::Mat<std::complex<float>> psiMatrix = calculate_psi_matrix(echoTimes, phiMatrix, fm, r2star);

                // Solve for water and fat

                for (int cha = 0; cha < CHA; cha++) {
                    arma::Mat<std::complex<float>> curWaterFat = arma::solve(psiMatrix, signals[cha]);
//                hesv(AhA, curWaterFat);
                    for (int k4 = 0; k4 < N; k4++) {
                        for (int k5 = 0; k5 < 2; k5++) { // 2 elements for water and fat currently
                            out(k1, k2, 0, cha, k4, k5, 0) = curWaterFat(k5, k4);
                        }
                    }
                }

            }
        }
        return out;
    }

    arma::Mat<std::complex<float>>
    calculatePhiMatrix(const FatWaterAlgorithm &a, float fieldStrength, const std::vector<float> &echoTimes,
                       uint16_t npeaks, uint16_t nspecies, uint16_t nte);

    hoNDArray<std::complex<float> >
    fatwater_separation(const hoNDArray<std::complex<float> > &data_orig, FatWaterParameters p,
                        FatWaterAlgorithm a) {

        GadgetronTimer timer("FatWater separation");

//        auto data = *downsample<std::complex<float>,2>(&data_orig);
        auto data = data_orig;
//        auto data = data_orig;
//        float sigma[] = {1,1,0,0,0,0,0};
//        filterGaussian(data,sigma);

        //Get some data parameters
        //7D, fixed order [X, Y, Z, CHA, N, S, LOC]
        uint16_t X = data.get_size(0);
        uint16_t Y = data.get_size(1);
        uint16_t Z = data.get_size(2);
        uint16_t CHA = data.get_size(3);
        uint16_t N = data.get_size(4);
        uint16_t S = data.get_size(5);
        uint16_t LOC = data.get_size(6);

        GDEBUG("Size of my array: %d, %d, %d .\n", X, Y, Z);


        float fieldStrength = p.fieldStrengthT_;
        std::vector<float> echoTimes = p.echoTimes_;
        bool precessionIsClockwise = p.precessionIsClockwise_;
        for (auto &te: echoTimes) {
            te = te * 0.001; // Echo times in seconds rather than milliseconds
        }

        GDEBUG("In toolbox - Field Strength: %f T \n", fieldStrength);
        for (auto &te: echoTimes) {
            GDEBUG("In toolbox - Echo time: %f seconds \n", te);
        }
        GDEBUG("In toolbox - PrecessionIsClockwise: %d \n", precessionIsClockwise);

        //Get or set some algorithm parameters
        Gadgetron::ChemicalSpecies w = a.species_[0];
        Gadgetron::ChemicalSpecies f = a.species_[1];

        GDEBUG("In toolbox - Fat peaks: %f  \n", f.ampFreq_[0].first);
        GDEBUG("In toolbox - Fat peaks 2: %f  \n", f.ampFreq_[0].second);




        auto average_fat_freq = std::accumulate(f.ampFreq_.begin(), f.ampFreq_.end(), 0if, [](auto val, auto tup) {
            return val + std::get<0>(tup) * std::get<1>(tup);
        }) /
                                std::accumulate(f.ampFreq_.begin(), f.ampFreq_.end(), 0.0if,
                                                [](auto val, auto tup) { return val + std::get<0>(tup); });
        average_fat_freq *= GAMMABAR * fieldStrength;





        // Set some initial parameters so we can get going
        // These will have to be specified in the XML file eventually
//        std::pair<float, float> range_r2star = std::make_pair(5.0, 500.0);
        std::pair<float, float> range_r2star = std::make_pair(0.0f, 0.0f);
        uint16_t num_r2star = 1;
        uint16_t num_r2star_fine = 1;
        std::pair<float, float> range_fm = std::make_pair(-500.0, 500.0);
        uint16_t num_fm = 200;
//        uint16_t num_iterations = num_fm*2;
        uint16_t num_iterations = 1;
        uint16_t subsample = 1;
        float lmap_power = 2.0;
        float lambda = 0.02;
        float lambda_extra = 0.0;

        //Check that we have reasonable data for fat-water separation


        //Calculate residual
        //

        uint16_t npeaks;
        uint16_t nspecies = a.species_.size();
        uint16_t nte = echoTimes.size();
        GDEBUG("In toolbox - NTE: %d \n", nte);

//	hoMatrix< std::complex<float> > phiMatrix(nte,nspecies);
        arma::Mat<std::complex<float>> phiMatrix = calculatePhiMatrix(a, fieldStrength, echoTimes, npeaks, nspecies,
                                                                      nte);


        std::vector<float> field_map_strengths(num_fm);
        field_map_strengths[0] = range_fm.first;
        for (int k1 = 1; k1 < num_fm; k1++) {
            field_map_strengths[k1] = range_fm.first + k1 * (range_fm.second - range_fm.first) / (num_fm - 1);
        }


        std::vector<float> r2stars(num_r2star);
        r2stars[0] = range_r2star.first;
        for (int k2 = 1; k2 < num_r2star; k2++) {
            r2stars[k2] = range_r2star.first + k2 * (range_r2star.second - range_r2star.first) / (num_r2star - 1);
        }

        std::vector<float> r2stars_fine(num_r2star_fine);
        r2stars_fine[0] = range_r2star.first;
        for (int k2 = 1; k2 < num_r2star_fine; k2++) {
            r2stars_fine[k2] =
                    range_r2star.first + k2 * (range_r2star.second - range_r2star.first) / (num_r2star_fine - 1);
        }


        auto Ps = CalculateResidualMap(echoTimes, num_r2star, num_fm, phiMatrix, field_map_strengths,
                                       r2stars);


        // Need to check that S = nte
        // N should be the number of contrasts (eg: for PSIR)
        hoNDArray<float> residual;
        hoNDArray<uint16_t> r2starIndex;
        CalculateResidualAndR2Star(num_r2star, num_fm, nte, Ps, data, residual, r2starIndex);




//        float sigma[] = {0,2,2};
//        filterGaussian(residual,sigma);

        auto dF = field_map_strengths[1] - field_map_strengths[0];
        auto dTE = echoTimes[1] - echoTimes[0];
        GDEBUG("Average fat freq %f dTE %f \n", std::abs(average_fat_freq), dF);
        hoNDArray<std::vector<uint16_t>> local_min_indices = find_local_minima(residual);
        hoNDArray<float> second_deriv = approx_second_derivative(residual, local_min_indices, dF);

//        second_deriv.fill(mean(&second_deriv));



        second_deriv += mean(&second_deriv) * lambda_extra;

//        second_deriv.fill(1.0);
        second_deriv *= lambda;
        second_deriv *= dF * dF;

//        enhance_regularization_edges(second_deriv,0.5);
//        smooth_edges(residual,0.1,0.01);
//        sqrt_inplace(&residual);

        hoNDArray<uint16_t> fmIndex = solve_MRF(num_iterations, field_map_strengths, residual, local_min_indices,
                                                second_deriv, std::abs(average_fat_freq), dF, dTE);

//        hoNDArray<uint16_t> fmIndex = solve_MRF_alphabeta(num_iterations, field_map_strengths, residual, second_deriv);


        auto r2star_map = calculate_r2star_map(data, fmIndex, r2stars_fine, field_map_strengths, phiMatrix, echoTimes);
//        auto r2star_map  = create_field_map(r2starIndex,r2stars);
        hoNDArray<float> field_map = create_field_map(fmIndex, field_map_strengths);


        // Do fat-water separation with current field map and R2* estimates
//        field_map = *upsample<float,2>(&field_map);
//        r2star_map = *upsample<float,2>(&r2star_map);


        auto out = separate_species(data_orig, echoTimes, phiMatrix, r2star_map, field_map);


        sqrt_inplace(&second_deriv);
//        second_deriv = *upsample<float,2>(&second_deriv);
//        second_deriv /= dF;
//
//        fat_water_mixed_fitting(field_map,r2star_map,out,data_orig,second_deriv, a,echoTimes,fieldStrength);


/*
        out = *upsample<std::complex<float>,2>(&out);
        field_map = *upsample<float,2>(&field_map);
        r2star_map = *upsample<float,2>(&r2star_map);
*/
//        fat_water_mixed_fitting(field_map, r2star_map, out, data_orig,second_deriv,
//                                a, echoTimes, fieldStrength);

        write_nd_array<float>(&field_map, "field_map.real");
        write_nd_array<float>(&r2star_map, "r2star_map.real");
        write_nd_array<float>(&residual, "residual_map.real");
        write_nd_array<float>(&second_deriv, "deriv.real");
        //Clean up as needed


        return out;
    }

    arma::Mat<std::complex<float>>
    calculatePhiMatrix(const FatWaterAlgorithm &a, float fieldStrength, const std::vector<float> &echoTimes,
                       uint16_t npeaks, uint16_t nspecies, uint16_t nte) {
        typedef arma::Mat<std::complex<float>> Cmat;
        Cmat phiMatrix = arma::zeros<Cmat>(nte, nspecies);
        for (int k1 = 0; k1 < nte; k1++) {
            for (int k2 = 0; k2 < nspecies; k2++) {
                npeaks = a.species_[k2].ampFreq_.size();
                for (int k3 = 0; k3 < npeaks; k3++) {
                    auto relAmp = a.species_[k2].ampFreq_[k3].first;
                    auto freq_hz = fieldStrength * GAMMABAR * a.species_[k2].ampFreq_[k3].second;
                    phiMatrix(k1, k2) += relAmp * exp(2if  * PI * echoTimes[k1] * freq_hz);

                }

            }
        }
        return phiMatrix;
    }


}
