#include "fatwater.h"

#include "hoMatrix.h"
#include "hoNDArray_linalg.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_reductions.h"
#include "hoArmadillo.h"

#include <boost/config.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/read_dimacs.hpp>
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
#include "simplexLagariaSolver.h"
#include "twoParaExpDecayOperator.h"
#include "curveFittingCostFunction.h"

#include <boost/random.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/math/constants/constants.hpp>
#include <armadillo>
#include <random>

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
    static constexpr float GAMMABAR = 42.576;
    static constexpr float PI = boost::math::constants::pi<float>();
    static std::mt19937 rng_state(4242);

    hoNDArray<uint16_t> create_field_map_proposal1(const hoNDArray<uint16_t>& field_map_index, const hoNDArray<std::vector<uint16_t>>& minima, const hoNDArray<float>& residuals, const std::vector<float> &field_map_strengths){

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        for (size_t i = 0; i < elements; i++){
            auto & mins  = minima[i];
            auto fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.begin(),mins.end(), [&](auto fqi){return fqi > fbi;}); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.end()) ? fbi : *fqmi;
        }

        return proposed_field_map_index;



    }

    hoNDArray<uint16_t> create_field_map_proposal2(const hoNDArray<uint16_t>& field_map_index, const hoNDArray<std::vector<uint16_t>>& minima, const hoNDArray<float>& residuals, const std::vector<float> &field_map_strengths){

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        for (size_t i = 0; i < elements; i++){
            auto & mins  = minima[i];
            auto fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.rbegin(),mins.rend(), [&](auto fqi){return fqi < fbi;}); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.rend()) ? fbi : *fqmi;
        }

        return proposed_field_map_index;


    }

    hoNDArray<uint16_t> create_field_map_proposal_standard(const hoNDArray<uint16_t>& field_map_index,int sign, uint16_t max_field_value){


        std::uniform_int_distribution<int> rng(1,3);

        int step_size = sign*rng(rng_state);
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        std::transform(field_map_index.begin(),field_map_index.end(),proposed_field_map_index.begin(),[=](uint16_t j){ return uint16_t(std::min(std::max(j+step_size,0),int(max_field_value)));});

        return proposed_field_map_index;
    }

    hoNDArray<std::vector<uint16_t>> find_local_minima(const hoNDArray<float>& residuals){

        const auto Y = residuals.get_size(2);
        const auto X = residuals.get_size(1);
        hoNDArray<std::vector<uint16_t>> result(X,Y);
        const auto steps = residuals.get_size(0);
        for (size_t k2 = 0; k2 < Y; k2++){
            for (size_t k1 = 0; k1 < X; k1++){
                float min = residuals(0,k1,k2);
                float max = min;
                std::vector<uint16_t> minima;
                for (size_t k0 = 0; k0 < steps; k0++){
                    min = std::min(min,residuals(k0,k1,k2));
                    max = std::max(max,residuals(k0,k1,k2));
                }

                for (size_t k0 = 1; k0 < steps-1; k0++){
                    if ((residuals(k0,k1,k2)-residuals(k0-1,k1,k2)) < 0 &&
                        (residuals(k0+1,k1,k2)-residuals(k0,k1,k2)) > 0 ){

                        //&&residuals(k0,k1,k2) < 0.3*(max-min)+min){
                        minima.push_back(k0);
                    }

                }
                auto comparator = [&](uint16_t i, uint16_t j){
                    return residuals(i,k1,k2) < residuals(j,k1,k2);
                };
//                GDEBUG("Minima size %i\n",minima.size());
//                std::sort(minima.begin(),minima.end(),comparator);

                result(k1,k2) = std::move(minima);
//                GDEBUG("K1 %i K2 %i \n",k1,k2);

            }
        }
        return result;
    }

    hoNDArray<float> approx_second_derivative(const hoNDArray<float> & residuals, const hoNDArray<uint16_t>& fmIndex, float step_size ){
        hoNDArray<float> second_deriv(fmIndex.get_dimensions());

        const auto Y = second_deriv.get_size(1);
        const auto X = second_deriv.get_size(0);

        for (uint16_t k2 = 0; k2 < Y; k2++) {
            for (uint16_t k1 = 0; k1 < X; k1++) {
                second_deriv(k1, k2) =
                        (residuals(fmIndex(k1, k2) - 1, k1, k2) + residuals(fmIndex(k1, k2) + 1, k1, k2) -
                         2 * residuals(fmIndex(k1, k2), k1, k2)) / (step_size * step_size);
            }
        }

        return second_deriv;


    }


    void add_regularization_edge(const hoNDArray <float> &field_map,
                                 const hoNDArray <float> &proposed_field_map, const size_t source_idx,
                                 const size_t sink_idx, std::vector<std::pair<size_t, size_t>> &edges,
                                 std::vector<float> &edge_weights, const size_t idx, const size_t idx2);

//    //Welcome to template Hell ala 1998. Enjoy.
//    typedef boost::adjacency_list<vecS, vecS, undirectedS> Traits;
//    typedef boost::adjacency_list<vecS, vecS, undirectedS,
//            boost::property<boost::vertex_color_t, boost::default_color_type,
//            boost::property<boost::vertex_predecessor_t, Traits::edge_descriptor,
//                    boost::property<boost::vertex_distance_t, float>>>,
//            boost::property<boost::edge_capacity_t,float,
//                    boost::property<boost::edge_residual_capacity_t,float,
//                            boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>> Graph;

//
    typedef boost::compressed_sparse_row_graph<bidirectionalS> Traits;

    typedef boost::compressed_sparse_row_graph<bidirectionalS,
            boost::property<boost::vertex_color_t, boost::default_color_type,
                    boost::property<boost::vertex_predecessor_t, Traits::edge_descriptor,
                            boost::property<boost::vertex_distance_t, float>>>,
            boost::property<boost::edge_capacity_t,float,
                    boost::property<boost::edge_residual_capacity_t,float,
                            boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>,no_property, std::size_t,std::size_t> Graph;


    void fix_reverse_edges(Graph &graph,const std::vector<size_t> dims,size_t source_idx,size_t sink_idx){

        auto edge_reverse_map = boost::get(boost::edge_reverse,graph);

        for (size_t k2 = 0; k2 < dims[1]; k2++) {
            for (size_t k1 = 0; k1 < dims[0]; k1++) {
                size_t idx = k2 * dims[0] + k1;

                if (k1 < (dims[0] - 1)) {
                    size_t idx2 = idx + 1;
                    auto edge = boost::edge(idx,idx2,graph).first;
                    auto reverse_edge = boost::edge(idx2,idx,graph).first;
                    edge_reverse_map[edge] = reverse_edge;
                    edge_reverse_map[reverse_edge] = edge;

                }

                if (k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0];
                    auto edge = boost::edge(idx,idx2,graph).first;
                    auto reverse_edge = boost::edge(idx2,idx,graph).first;
                    edge_reverse_map[edge] = reverse_edge;
                    edge_reverse_map[reverse_edge] = edge;
                }

                if (k1 < (dims[0] - 1) && k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0] + 1;
                    auto edge = boost::edge(idx,idx2,graph).first;
                    auto reverse_edge = boost::edge(idx2,idx,graph).first;
                    edge_reverse_map[edge] = reverse_edge;
                    edge_reverse_map[reverse_edge] = edge;
                }

                auto edge = boost::edge(idx,source_idx,graph).first;
                auto reverse_edge = boost::edge(source_idx,idx,graph).first;
                edge_reverse_map[edge] = reverse_edge;
                edge_reverse_map[reverse_edge] = edge;

                edge = boost::edge(idx,sink_idx,graph).first;
                reverse_edge = boost::edge(sink_idx,idx,graph).first;
                edge_reverse_map[edge] = reverse_edge;
                edge_reverse_map[reverse_edge] = edge;
            }
        }

    }

    Graph create_empty_graph(const hoNDArray<float> &field_map) {
        std::vector<std::pair<size_t, size_t>> edges;



        const auto dims = *field_map.get_dimensions();

        const size_t source_idx = field_map.get_number_of_elements();
        const size_t sink_idx = source_idx + 1;

        for (size_t k2 = 0; k2 < dims[1]; k2++) {
            for (size_t k1 = 0; k1 < dims[0]; k1++) {
                size_t idx = k2 * dims[0] + k1;

                if (k1 < (dims[0] - 1)) {
                    size_t idx2 = idx + 1;
                    edges.emplace_back(idx, idx2);
                    edges.emplace_back(idx2, idx);
                }

                if (k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0];
                    edges.emplace_back(idx, idx2);
                    edges.emplace_back(idx2, idx);
                }

                if (k1 < (dims[0] - 1) && k2 < (dims[1] - 1)) {
                    size_t idx2 = idx + dims[0] + 1;
                    edges.emplace_back(idx, idx2);
                    edges.emplace_back(idx2, idx);
                }

                edges.emplace_back(source_idx, idx);
                edges.emplace_back(idx, source_idx);

                edges.emplace_back(sink_idx, idx);
                edges.emplace_back(idx, sink_idx);

            }
        }
        struct constant_iterator : public std::iterator<std::forward_iterator_tag,float>  {
        public:
            constant_iterator& operator++(){ return *this;}
            float operator*(){ return 0;}
        };


        Graph  graph(boost::edges_are_unsorted_multi_pass_t (),edges.begin(),edges.end(), constant_iterator(),field_map.get_number_of_elements()+2);

        fix_reverse_edges(graph,dims,source_idx,sink_idx);
        return graph;

    }


    void add_to_edge(size_t idx, size_t idx2, float value, Graph& graph){
        auto edge_capacity_map = boost::get(boost::edge_capacity,graph);
        auto edge = boost::edge(idx,idx2,graph);
        edge_capacity_map[edge.first] += value;

        edge = boost::edge(idx2,idx,graph);
        edge_capacity_map[edge.first] += value;

    }

    void update_regularization_edge(Graph& graph, const hoNDArray<float> &field_map,
                                    const hoNDArray<float> &proposed_field_map, const hoNDArray<float> second_deriv,
                                    const size_t source_idx,
                                    const size_t sink_idx, const size_t idx, const size_t idx2) {

        auto f_value1 = field_map[idx];
        auto pf_value1 = proposed_field_map[idx];
        auto f_value2 = field_map[idx2];
        auto pf_value2 = proposed_field_map[idx2];
        float weight = std::norm(pf_value1 - f_value2) + std::norm(f_value1 - pf_value2)
                       - std::norm(f_value1 - f_value2) - std::norm(pf_value1 - pf_value2);

        weight *= std::min(second_deriv[idx],second_deriv[idx2]);

        add_to_edge(idx,idx2,weight,graph);
        float aq = std::norm(pf_value1 - f_value2) - std::norm(f_value1 - f_value2);

        if (aq > 0){
            add_to_edge(source_idx,idx,aq,graph);
        } else {
            add_to_edge(idx,sink_idx,-aq,graph);
        }

        float aj = std::norm(f_value1 - pf_value2) - std::norm(f_value1 - f_value2);
        if (aj > 0){
            add_to_edge(source_idx,idx2,aj,graph);
        } else {
            add_to_edge(idx2,sink_idx,-aj,graph);
        }

    }





    Graph make_graph(const hoNDArray<float> &field_map, const hoNDArray<float> &proposed_field_map,
                         const hoNDArray<float> &residual_diff_map, const hoNDArray<float> &second_deriv) {

        const auto dims = *field_map.get_dimensions();



        const size_t source_idx = field_map.get_number_of_elements();
        const size_t sink_idx = source_idx+1;

        Graph graph = create_empty_graph(field_map);

        //Add regularization edges
        for (size_t k2 = 0; k2 < dims[1]; k2++){
            for (size_t k1 = 0; k1 < dims[0]; k1++){
                size_t idx = k2*dims[0]+k1;

                if (k1 < (dims[0]-1)){
                    size_t idx2 = idx+1;
                    update_regularization_edge(graph,field_map,proposed_field_map,second_deriv, source_idx,sink_idx,idx,idx2);
                }


                if (k2 < (dims[1]-1)){
                    size_t idx2 = idx + dims[0];
                    update_regularization_edge(graph,field_map,proposed_field_map,second_deriv, source_idx,sink_idx,idx,idx2);
                }
                if (k1 < (dims[0]-1) && k2 < (dims[1]-1)){
                    size_t idx2 = idx+dims[0]+1;
                    update_regularization_edge(graph,field_map,proposed_field_map,second_deriv, source_idx,sink_idx,idx,idx2);
                }


                float residual_diff = residual_diff_map[idx];

                if (residual_diff > 0){
                    add_to_edge(source_idx,idx,residual_diff,graph);
                } else {
                    add_to_edge(idx,sink_idx,-residual_diff,graph);
                }

            }
        }

        return graph;
    }

    hoNDArray<float> create_field_map(const hoNDArray<uint16_t>& field_map_index, const std::vector<float>& field_map_strengths){
        hoNDArray<float> field_map(field_map_index.get_dimensions());
        std::transform(field_map_index.begin(),field_map_index.end(),field_map.begin(),[&](uint16_t i ){return field_map_strengths[i]; });
        return field_map;
    }


    void update_field_map(hoNDArray<uint16_t> &field_map_index, const hoNDArray<uint16_t> &proposed_field_map_index,
                          const hoNDArray<float> &residuals_map, const hoNDArray<float>& second_deriv, std::vector<float> field_map_strengths) {

        hoNDArray<float> field_map = create_field_map(field_map_index, field_map_strengths);
        hoNDArray<float> proposed_field_map = create_field_map(proposed_field_map_index,field_map_strengths);

        hoNDArray<float> residual_diff_map(field_map.get_dimensions());
        const auto X = field_map.get_size(0);
        const auto Y = field_map.get_size(1);

        for (size_t k2 = 0; k2 < Y; k2++){
            for(size_t k1 = 0; k1 < X; k1++){
                residual_diff_map(k1,k2) = residuals_map(field_map_index(k1,k2),k1,k2) -
                                           residuals_map(proposed_field_map_index(k1,k2),k1,k2);
            }
        }



        Graph graph = make_graph(field_map, proposed_field_map, residual_diff_map, second_deriv);
        size_t source_idx = field_map.get_number_of_elements();
        size_t sink_idx = source_idx+1;
//        auto parities = boost::make_one_bit_color_map(num_vertices(graph), get(boost::vertex_index, graph));
//
        // run the Stoer-Wagner algorithm to obtain the min-cut weight. `parities` is also filled in.
        // This is
//        boost::stoer_wagner_min_cut(graph, boost::get(boost::edge_weight, graph), boost::parity_map(parities));
        Graph::vertex_descriptor source = boost::vertex(source_idx,graph);
        Graph::vertex_descriptor sink = boost::vertex(sink_idx,graph);

        boost::boykov_kolmogorov_max_flow(graph,source,sink);

        auto color_map = boost::get(vertex_color,graph);

        // Ok, let's figure out what labels were assigned to the source.
        auto source_label = boost::get(color_map,source_idx);
        //And update the field_map
        for (size_t i = 0; i < field_map.get_number_of_elements(); i++){
            if (boost::get(color_map,i) == source_label)
                field_map_index[i] = proposed_field_map_index[i];
        }

    }
//
//    void add_regularization_edge(const hoNDArray<float> &field_map,
//                                 const hoNDArray<float> &proposed_field_map, const size_t source_idx,
//                                 const size_t sink_idx, std::vector<std::pair<size_t, size_t>> &edges,
//                                 std::vector<float> &edge_weights, const size_t idx, const size_t idx2) {
//        edges.emplace_back(idx, idx2);
//        auto f_value1 = field_map[idx];
//        auto pf_value1 = proposed_field_map[idx];
//        auto f_value2 = field_map[idx2];
//        auto pf_value2 = proposed_field_map[idx2];
//        float weight = std::norm(pf_value1 - f_value2) + std::norm(f_value1 - pf_value2)
//                       - std::norm(f_value1 - f_value2) - std::norm(pf_value1 - pf_value2);
//        edge_weights.push_back(weight);
//
//        float aq = std::norm(pf_value1 - f_value2) - std::norm(f_value1 - f_value2);
//
//        if (aq > 0){
//            edges.emplace_back(source_idx,idx);
//            edge_weights.push_back(aq);
//        } else {
//            edges.emplace_back(idx,sink_idx);
//            edge_weights.push_back(-aq);
//        }
//
//        float aj = std::norm(f_value1 - pf_value2) - std::norm(f_value1 - f_value2);
//        if (aj > 0){
//            edges.emplace_back(source_idx,idx2);
//            edge_weights.push_back(aj);
//
//        } else {
//            edges.emplace_back(idx2,sink_idx);
//            edge_weights.push_back(-aj);
//        }
//
//
//
//
//    }

    hoNDArray <std::complex<float>>
    CalculateResidualMap(const std::vector<float> &echoTimes, uint16_t num_r2star, uint16_t num_fm, uint16_t nspecies,
                         uint16_t nte, const arma::Mat<std::complex<float>> &phiMatrix,
                         const std::vector<float> &field_map_strengths, const std::vector<float> &r2stars);

    hoNDArray<std::complex<float> > fatwater_separation(hoNDArray<std::complex<float> > &data, FatWaterParameters p,
                                                        FatWaterAlgorithm a) {

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

        hoNDArray<std::complex<float> > out(X, Y, Z, CHA, N, 2, LOC); // S dimension gets replaced by water/fat stuff

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
        //Gadgetron::ChemicalSpecies w = a.species_[0];
        //Gadgetron::ChemicalSpecies f = a.species_[1];

        //	GDEBUG("In toolbox - Fat peaks: %f  \n", f.ampFreq_[0].first);
        //	GDEBUG("In toolbox - Fat peaks 2: %f  \n", f.ampFreq_[0].second);

        // Set some initial parameters so we can get going
        // These will have to be specified in the XML file eventually
        std::pair<float, float> range_r2star = std::make_pair(0.0, 0.0);
        uint16_t num_r2star = 1;
        std::pair<float, float> range_fm = std::make_pair(-400.0, 400.0);
        uint16_t num_fm = 201;
        uint16_t num_iterations = 40;
        uint16_t subsample = 1;
        float lmap_power = 2.0;
        float lambda = 0.02;
        float lambda_extra = 0.02;

        //Check that we have reasonable data for fat-water separation


        //Calculate residual
        //
        float relAmp, freq_hz;
        uint16_t npeaks;
        uint16_t nspecies = a.species_.size();
        uint16_t nte = echoTimes.size();
        GDEBUG("In toolbox - NTE: %d \n", nte);

//	hoMatrix< std::complex<float> > phiMatrix(nte,nspecies);
        typedef arma::Mat<std::complex<float>> Cmat;
        Cmat phiMatrix = arma::zeros<Cmat>(nte, nspecies);
        for (int k1 = 0; k1 < nte; k1++) {
            for (int k2 = 0; k2 < nspecies; k2++) {
                npeaks = a.species_[k2].ampFreq_.size();
                for (int k3 = 0; k3 < npeaks; k3++) {
                    relAmp = a.species_[k2].ampFreq_[k3].first;
                    freq_hz = fieldStrength * GAMMABAR * a.species_[k2].ampFreq_[k3].second;
                    phiMatrix(k1, k2) += relAmp * std::complex<float>(cos(2 * PI * echoTimes[k1] * freq_hz),
                                                                      sin(2 * PI * echoTimes[k1] * freq_hz));
                }
                GDEBUG("Cur value phiMatrix = (%f,%f) \n", phiMatrix(k1, k2).real(), phiMatrix(k1, k2).imag());
            }
        }



        float fm;
        std::vector<float> field_map_strengths(num_fm);
        field_map_strengths[0] = range_fm.first;
        for (int k1 = 1; k1 < num_fm; k1++) {
            field_map_strengths[k1] = range_fm.first + k1 * (range_fm.second - range_fm.first) / (num_fm - 1);
        }

        float r2star;
        std::vector<float> r2stars(num_r2star);
        r2stars[0] = range_r2star.first;
        for (int k2 = 1; k2 < num_r2star; k2++) {
            r2stars[k2] = range_r2star.first + k2 * (range_r2star.second - range_r2star.first) / (num_r2star - 1);
        }






        GDEBUG("Calculating residiaul map");
        auto Ps = CalculateResidualMap(echoTimes, num_r2star, num_fm, nspecies, nte, phiMatrix, field_map_strengths,
                                       r2stars);


        // Need to check that S = nte
        // N should be the number of contrasts (eg: for PSIR)
        hoMatrix<std::complex<float> > tempResVector(S, N);
        Cmat tempSignal(S, N);
        hoNDArray<float> residual(num_fm, X, Y);
        hoNDArray<uint16_t> r2starIndex(X, Y, num_fm);
        hoNDArray<uint16_t> fmIndex(X, Y);
        float curResidual, minResidual, minResidual2;
        Cmat P(nte,nte);
        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {
                // Get current signal
                for (int k4 = 0; k4 < N; k4++) {
                    for (int k5 = 0; k5 < S; k5++) {
                        tempSignal(k5, k4) = data(k1, k2, 0, 0, k4, k5, 0);
                        if (k1 == 107 && k2 == 144) {
                            tempSignal(k5, k4) = std::complex<float>(1000.0, 0.0);;
                            GDEBUG(" (%d,%d) -->  %f + i %f \n", k5, k4, tempSignal(k5, k4).real(),
                                   tempSignal(k5, k4).imag());
                        }

                    }
                }

                minResidual2 = 1.0 + arma::norm(tempSignal);

                for (int k3 = 0; k3 < num_fm; k3++) {

                    minResidual = 1.0 + arma::norm(tempSignal);

                    for (int k4 = 0; k4 < num_r2star; k4++) {
                        // Get current projector matrix
                        for (int k5 = 0; k5 < nte; k5++) {
                            for (int k6 = 0; k6 < nte; k6++) {
                                P(k5, k6) = Ps(k5, k6, k3, k4);
                            }
                        }

                        // Apply projector
//                        gemm(tempResVector, P, false, tempSignal, false);


                        curResidual = arma::norm(P*tempSignal);

                        if (curResidual < minResidual) {
                            minResidual = curResidual;
                            r2starIndex(k1, k2, k3) = k4;
                        }
                    }
                    residual(k3, k1, k2) = minResidual;

                    if (minResidual < minResidual2) {
                        minResidual2 = minResidual;
                        fmIndex(k1, k2) = k3;
                    }

                    if (k1 == 107 && k2 == 144) {
                        GDEBUG(" %f -->  %f \n", field_map_strengths[k3], minResidual);
                    }
                }
            }
        }


        GDEBUG("Second derivative \n");
        hoNDArray<float> second_deriv = approx_second_derivative(residual,fmIndex,field_map_strengths[1]-field_map_strengths[0]);
        GDEBUG("Finding local minima \n");
        hoNDArray<std::vector<uint16_t>> local_min_indices = find_local_minima(residual);
        GDEBUG("Aaaand, done");

        std::uniform_int_distribution<int> coinflip(0,1);

        auto fmIndex_update = fmIndex;
        for (int i = 0; i < num_iterations; i++){
            GDEBUG("Iteration number %i \n", i);
            if ( coinflip(rng_state)  || i < 15){
                if (coinflip(rng_state)){
                    fmIndex_update = create_field_map_proposal1(fmIndex,local_min_indices,residual,field_map_strengths);
                } else {
                    fmIndex_update = create_field_map_proposal2(fmIndex,local_min_indices,residual,field_map_strengths);
                }
            } else {
                fmIndex_update = create_field_map_proposal_standard(fmIndex,std::pow(-1,i),field_map_strengths.size());
            }
            GDEBUG("Proposal created");
            update_field_map(fmIndex,fmIndex_update,residual,second_deriv,field_map_strengths);
        }



        hoNDArray<float> field_map = create_field_map(fmIndex,field_map_strengths);



        //Do final calculations once the field map is done
//        hoMatrix<std::complex<float> > curWaterFat(2, N);
//        hoMatrix<std::complex<float> > AhA(2, 2);
        // Do fat-water separation with current field map and R2* estimates
        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {

                Cmat tempSignal(S,N);

                // Get current signal
                for (int k4 = 0; k4 < N; k4++) {
                    for (int k5 = 0; k5 < S; k5++) {
                        tempSignal(k5, k4) = data(k1, k2, 0, 0, k4, k5, 0);
                    }
                }
                // Get current Psi matrix
//                fm = field_map_strengths[fmIndex(k1, k2)];
                fm = field_map(k1,k2);
                r2star = r2stars[r2starIndex(k1, k2, fmIndex(k1, k2))];
                Cmat psiMatrix(nte,nspecies);
                for (int k3 = 0; k3 < nte; k3++) {
                    auto curModulation = exp(-r2star * echoTimes[k3]) * std::complex<float>(cos(2 * PI * echoTimes[k3] * fm),
                                                                                            sin(2 * PI * echoTimes[k3] *
                                                                                                fm));
                    for (int k4 = 0; k4 < nspecies; k4++) {
                        psiMatrix(k3, k4) = phiMatrix(k3, k4) * curModulation;
                    }
                }

                // Solve for water and fat



                Cmat curWaterFat = arma::solve(psiMatrix,tempSignal);
//                hesv(AhA, curWaterFat);
                for (int k4 = 0; k4 < N; k4++) {
                    for (int k5 = 0; k5 < 2; k5++) { // 2 elements for water and fat currently
                        out(k1, k2, 0, 0, k4, k5, 0) = curWaterFat(k5, k4);
                    }
                }

            }
        }



        //Clean up as needed


        return out;
    }

    hoNDArray <std::complex<float>>
    CalculateResidualMap(const std::vector<float> &echoTimes, uint16_t num_r2star, uint16_t num_fm, uint16_t nspecies,
                         uint16_t nte, const arma::Mat<std::complex<float>> &phiMatrix,
                         const std::vector<float> &field_map_strengths, const std::vector<float> &r2stars) {
        arma::Mat<std::complex<float>> psiMatrix(nte, nspecies);
        hoNDArray<std::complex<float> > Ps(nte, nte, num_fm, num_r2star);
        arma::Mat<std::complex<float>> P(nte, nte);

        for (int k3 = 0; k3 < num_fm; k3++) {
            float fm = field_map_strengths[k3];
            for (int k4 = 0; k4 < num_r2star; k4++) {
                float r2star = r2stars[k4];


                for (int k1 = 0; k1 < nte; k1++) {
                    auto curModulation = exp(-r2star * echoTimes[k1]) * std::complex<float>(cos(2 * PI * echoTimes[k1] * fm),
                                                                                            sin(2 * PI * echoTimes[k1] *
                                                                                                fm));
                    for (int k2 = 0; k2 < nspecies; k2++) {
                        psiMatrix(k1, k2) = phiMatrix(k1,k2)*curModulation;
                    }
                }
                arma::Mat<std::complex<float>> P = arma::eye<arma::Mat<std::complex<float>>>(nte, nte) - psiMatrix *
                                                                                                         arma::solve(psiMatrix.t() * psiMatrix, psiMatrix.t());

// Keep all projector matrices together
                for (int k1 = 0; k1 < nte; k1++) {
                    for (int k2 = 0; k2 < nte; k2++) {
                        Ps(k1, k2, k3, k4) = P(k1, k2);
                    }
                }
            }
        }
        return Ps;
    }
}
