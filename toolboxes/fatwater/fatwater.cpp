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
#include "simplexLagariaSolver.h"
#include "twoParaExpDecayOperator.h"
#include "curveFittingCostFunction.h"

#include <boost/random.hpp>
#include <boost/math/constants/constants.hpp>
#include <armadillo>
#include <random>
#include <cpu/hoNDArray_fileio.h>
#include <GadgetronTimer.h>
#include <complex>
#include "FatWaterMixedFitting.h"

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

    hoNDArray<uint16_t> create_field_map_proposal1(const hoNDArray<uint16_t>& field_map_index, const hoNDArray<std::vector<uint16_t>>& minima, const hoNDArray<float>& residuals, const std::vector<float> &field_map_strengths){

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        for (size_t i = 0; i < elements; i++){
            auto & mins  = minima[i];
            auto fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.begin(),mins.end(), [&](auto fqi){return fqi > (fbi+20);}); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.end()) ? fbi+1 : *fqmi;
        }

        return proposed_field_map_index;



    }

    hoNDArray<uint16_t> create_field_map_proposal2(const hoNDArray<uint16_t>& field_map_index, const hoNDArray<std::vector<uint16_t>>& minima, const hoNDArray<float>& residuals, const std::vector<float> &field_map_strengths){

        const size_t elements = field_map_index.get_number_of_elements();
        hoNDArray<uint16_t> proposed_field_map_index(field_map_index.get_dimensions());
        const size_t field_maps = field_map_strengths.size();
        for (size_t i = 0; i < elements; i++){
            auto & mins  = minima[i];
            auto fbi = field_map_index[i];
            auto fqmi = std::find_if(mins.rbegin(),mins.rend(), [&](auto fqi){return fqi < (fbi-20);}); //Find smallest
            proposed_field_map_index[i] = (fqmi == mins.rend()) ? std::max(fbi,uint16_t(1))-1 : *fqmi;
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

    hoNDArray<std::vector<uint16_t>> find_local_minima(const hoNDArray<float>& residuals,float threshold = 0.06f){


        auto threshold_signal = std::move(*sum(&residuals,0));
        sqrt_inplace(&threshold_signal);
        threshold_signal /= max(&threshold_signal);


        const auto Y = residuals.get_size(2);
        const auto X = residuals.get_size(1);
        hoNDArray<std::vector<uint16_t>> result(X,Y);
        const auto steps = residuals.get_size(0);
        for (size_t k2 = 0; k2 < Y; k2++){
            for (size_t k1 = 0; k1 < X; k1++){

                std::vector<uint16_t> minima;
                if (threshold_signal(k1,k2) > threshold) {
                    for (size_t k0 = 1; k0 < steps - 1; k0++) {
                        if ((residuals(k0, k1, k2) - residuals(k0 - 1, k1, k2)) < 0 &&
                            (residuals(k0 + 1, k1, k2) - residuals(k0, k1, k2)) > 0) {
                            minima.push_back(k0);
                        }

                    }
                }
                result(k1,k2) = std::move(minima);
            }
        }
        return result;
    }

    hoNDArray<float> approx_second_derivative(const hoNDArray<float> & residuals, const hoNDArray<std::vector<uint16_t>>& local_min_indices, float step_size ){
        hoNDArray<float> second_deriv(local_min_indices.get_dimensions());

        const auto Y = second_deriv.get_size(1);
        const auto X = second_deriv.get_size(0);
        const auto nfields = residuals.get_size(0);

        for (uint16_t k2 = 0; k2 < Y; k2++) {
            for (uint16_t k1 = 0; k1 < X; k1++) {


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

                    auto sd  =
                            (residuals(minimum - 1, k1, k2) + residuals(minimum + 1, k1, k2) -
                             2 * residuals(minimum, k1, k2)) / (step_size * step_size);
                    second_deriv(k1, k2) = std::max(sd,0.0f);


            }
        }

//        second_deriv.fill(1.0f);

        return second_deriv;


    }




    typedef ImageGraph Graph;



    void update_regularization_edge(Graph &graph, const hoNDArray<float> &field_map, const hoNDArray<float> &proposed_field_map,
                                        const hoNDArray<float> &second_deriv, const size_t idx, const size_t idx2,
                                        const size_t edge_idx, float scaling) {

        auto f_value1 = field_map[idx];
        auto pf_value1 = proposed_field_map[idx];
        auto f_value2 = field_map[idx2];
        auto pf_value2 = proposed_field_map[idx2];
        float weight = std::norm(pf_value1 - f_value2) + std::norm(f_value1 - pf_value2)
                       - std::norm(f_value1 - f_value2) - std::norm(pf_value1 - pf_value2);

        assert(weight >= 0);

        float lambda = std::max(std::min(second_deriv[idx],second_deriv[idx2]),0.0f);
        weight *= lambda*scaling;

        assert(lambda >= 0);

        auto& capacity_map = graph.edge_capacity_map;

        capacity_map[edge_idx] += weight;
        capacity_map[graph.reverse(edge_idx)] += weight;

        {
            float aq = lambda * (std::norm(pf_value1 - f_value2) - std::norm(f_value1 - f_value2));

            if (aq > 0) {
                capacity_map[graph.edge_from_source(idx)] += aq;
//            capacity_map[graph.edge_to_source(idx)] += aq;

            } else {
//            capacity_map[graph.edge_from_sink(idx)] -= aq;
                capacity_map[graph.edge_to_sink(idx)] -= aq;
            }
        }

        {
            float aj = lambda * (std::norm(f_value1 - pf_value2) - std::norm(f_value1 - f_value2));
            if (aj > 0) {
                capacity_map[graph.edge_from_source(idx2)] += aj;
//            capacity_map[graph.edge_to_source(idx2)] += aj;

            } else {
//            capacity_map[graph.edge_from_sink(idx2)] -= aj;
                capacity_map[graph.edge_to_sink(idx2)] -= aj;
            }
        }


    }





    Graph make_graph(const hoNDArray<float> &field_map, const hoNDArray<float> &proposed_field_map,
                         const hoNDArray<float> &residual_diff_map, const hoNDArray<float> &second_deriv) {

        const auto dims = *field_map.get_dimensions();



        const size_t source_idx = field_map.get_number_of_elements();
        const size_t sink_idx = source_idx+1;

        Graph graph = Graph(dims[0],dims[1]);

        auto& capacity_map = graph.edge_capacity_map;
        //Add regularization edges
        for (size_t k2 = 0; k2 < dims[1]; k2++){
            for (size_t k1 = 0; k1 < dims[0]; k1++){
                size_t idx = k2*dims[0]+k1;



                if (k1 < (dims[0]-1)){
                    size_t idx2 = idx+1;
                    size_t edge = graph.edge_from_offset(idx,vector_td<int,2>(1,0));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,1);
                }


                if (k2 < (dims[1]-1)){
                    size_t idx2 = idx + dims[0];
                    size_t edge = graph.edge_from_offset(idx,vector_td<int,2>(0,1));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,1);
                }
                if (k1 < (dims[0]-1) && k2 < (dims[1]-1)){
                    size_t idx2 = idx+dims[0]+1;
                    size_t edge = graph.edge_from_offset(idx,vector_td<int,2>(1,1));
                    update_regularization_edge(graph, field_map, proposed_field_map, second_deriv, idx, idx2, edge,1/std::sqrt(2.0f));
                }


                float residual_diff = residual_diff_map[idx];

                if (residual_diff > 0){
                     capacity_map[graph.edge_from_source(idx)] += residual_diff;

                } else {
                    capacity_map[graph.edge_to_sink(idx)] -= residual_diff;
                }

            }
        }

        return graph;
    }

    hoNDArray<float> create_field_map(const hoNDArray<uint16_t>& field_map_index, const std::vector<float>& field_map_strengths){
        const uint16_t max_val = field_map_strengths.size()-1;
        hoNDArray<float> field_map(field_map_index.get_dimensions());
        std::transform(field_map_index.begin(),field_map_index.end(),field_map.begin(),[&](uint16_t i ){return field_map_strengths[std::min(i,max_val)]; });
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
                residual_diff_map(k1,k2) = residuals_map(field_map_index(k1,k2),k1,k2)-
                                           residuals_map(proposed_field_map_index(k1,k2),k1,k2);


            }
        }



        Graph graph = make_graph(field_map, proposed_field_map, residual_diff_map, second_deriv);

        Graph::vertex_descriptor source = graph.source_vertex;
        Graph::vertex_descriptor sink = graph.sink_vertex;

        float flow = boost::boykov_kolmogorov_max_flow(graph,source,sink);

        auto color_map = boost::get(vertex_color,graph);

        // Ok, let's figure out what labels were assigned to the source.
        auto source_label = boost::get(color_map,source);

        //And update the field_map
        size_t updated_voxels = 0;
        for (size_t i = 0; i < field_map.get_number_of_elements(); i++){
            if (boost::get(color_map,i) == boost::default_color_type::black_color) {
                updated_voxels++;
                field_map_index[i] = proposed_field_map_index[i];
            }
        }


    }
//

    hoNDArray <std::complex<float>>
    CalculateResidualMap(const std::vector<float> &echoTimes, uint16_t num_r2star, uint16_t num_fm, uint16_t nspecies,
                         uint16_t nte, const arma::Mat<std::complex<float>> &phiMatrix,
                         const std::vector<float> &field_map_strengths, const std::vector<float> &r2stars) {

        hoNDArray<std::complex<float> > Ps(nte, nte, num_fm, num_r2star);

#pragma omp parallel for collapse(2)
        for (int k3 = 0; k3 < num_fm; k3++) {
            for (int k4 = 0; k4 < num_r2star; k4++) {
                float fm = field_map_strengths[k3];
                float r2star = r2stars[k4];

                arma::Mat<std::complex<float>> psiMatrix(nte, nspecies);
                for (int k1 = 0; k1 < nte; k1++) {
                    auto curModulation = exp(-r2star * echoTimes[k1]+ 2if * PI * echoTimes[k1] * fm);
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
    hoNDArray<std::complex<float> > fatwater_separation(const hoNDArray<std::complex<float> > &data_orig, FatWaterParameters p,
                                                        FatWaterAlgorithm a) {

        GadgetronTimer timer("FatWater separation");

        auto data = *downsample<std::complex<float>,2>(&data_orig);
//        auto data = data_orig;
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
        Gadgetron::ChemicalSpecies w = a.species_[0];
        Gadgetron::ChemicalSpecies f = a.species_[1];

        	GDEBUG("In toolbox - Fat peaks: %f  \n", f.ampFreq_[0].first);
        	GDEBUG("In toolbox - Fat peaks 2: %f  \n", f.ampFreq_[0].second);

        // Set some initial parameters so we can get going
        // These will have to be specified in the XML file eventually
        std::pair<float, float> range_r2star = std::make_pair(5.0, 500.0);
        uint16_t num_r2star = 5;
        std::pair<float, float> range_fm = std::make_pair(-500.0, 500.0);
        uint16_t num_fm = 201;
        uint16_t num_iterations = 60;
        uint16_t subsample = 1;
        float lmap_power = 2.0;
        float lambda = 0.02;
        float lambda_extra = 0.01;

        //Check that we have reasonable data for fat-water separation


        //Calculate residual
        //

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
                    auto relAmp = a.species_[k2].ampFreq_[k3].first;
                    auto freq_hz = fieldStrength * GAMMABAR * a.species_[k2].ampFreq_[k3].second;
                    phiMatrix(k1, k2) += relAmp * exp(2if * PI * echoTimes[k1] * freq_hz);

                }

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







        auto Ps = CalculateResidualMap(echoTimes, num_r2star, num_fm, nspecies, nte, phiMatrix, field_map_strengths,
                                       r2stars);


        // Need to check that S = nte
        // N should be the number of contrasts (eg: for PSIR)
        hoMatrix<std::complex<float> > tempResVector(S, N);

        hoNDArray<float> residual(num_fm, X, Y);
        hoNDArray<uint16_t> r2starIndex(X, Y, num_fm);


#pragma omp parallel for collapse(2)
        for (int k1 = 0; k1 < X; k1++) {
            for (int k2 = 0; k2 < Y; k2++) {
                // Get current signal
                Cmat tempSignal(S, N);
                for (int k4 = 0; k4 < N; k4++) {
                    for (int k5 = 0; k5 < S; k5++) {
                        tempSignal(k5, k4) = data(k1, k2, 0, 0, k4, k5, 0);

                    }
                }


                for (int k3 = 0; k3 < num_fm; k3++) {

                    float minResidual = std::numeric_limits<float>::max();

                    for (int k4 = 0; k4 < num_r2star; k4++) {
                        Cmat P(nte,nte);
                        // Get current projector matrix
                        for (int k5 = 0; k5 < nte; k5++) {
                            for (int k6 = 0; k6 < nte; k6++) {
                                P(k5, k6) = Ps(k5, k6, k3, k4);
                            }
                        }

                        // Apply projector


                        Cmat projected = P*tempSignal;
                        float curResidual = std::accumulate(projected.begin(),projected.end(),0.0f,[](auto v1,auto v2){ return v1+std::norm(v2);});

                        if (curResidual < minResidual) {
                            minResidual = curResidual;
                            r2starIndex(k1, k2, k3) = k4;
                        }
                    }
                    residual(k3, k1, k2) = minResidual;

                }
            }
        }



        hoNDArray<std::vector<uint16_t>> local_min_indices = find_local_minima(residual);
        hoNDArray<float> second_deriv = approx_second_derivative(residual,local_min_indices,field_map_strengths[1]-field_map_strengths[0]);

        second_deriv.fill(mean(&second_deriv));



        second_deriv += mean(&second_deriv)*lambda_extra;

        second_deriv *= lambda;


        std::uniform_int_distribution<int> coinflip(0,1);
        hoNDArray<uint16_t> fmIndex(X, Y);
        fmIndex.fill(num_fm/2);

        auto fmIndex_update = fmIndex;

        for (int i = 0; i < num_iterations; i++){

            if ( coinflip(rng_state)  || i < 15){
                if (i%2){
                    fmIndex_update = create_field_map_proposal1(fmIndex,local_min_indices,residual,field_map_strengths);
                } else {
                    fmIndex_update = create_field_map_proposal2(fmIndex,local_min_indices,residual,field_map_strengths);
                }
            } else {
                fmIndex_update = create_field_map_proposal_standard(fmIndex,std::pow(-1,i),field_map_strengths.size());
            }

            update_field_map(fmIndex,fmIndex_update,residual,second_deriv,field_map_strengths);


        }
        {
            hoNDArray<float> field_map_update = create_field_map(fmIndex_update, field_map_strengths);
        }

        hoNDArray<float> field_map = create_field_map(fmIndex,field_map_strengths);


        // Do fat-water separation with current field map and R2* estimates
        hoNDArray<float> r2star_map(field_map.get_dimensions());

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
                r2star_map(k1,k2) = r2star;
                Cmat psiMatrix(nte,nspecies);
                for (int k3 = 0; k3 < nte; k3++) {
                    auto curModulation = exp(-r2star * echoTimes[k3] +2if* PI * echoTimes[k3] * fm);
                    for (int k4 = 0; k4 < nspecies; k4++) {
                        psiMatrix(k3, k4) = phiMatrix(k3, k4) * curModulation;
                    }
                }

                // Solve for water and fat



                Cmat curWaterFat = arma::solve(psiMatrix.t()*psiMatrix,psiMatrix.t()*tempSignal,arma::solve_opts::equilibrate);
//                hesv(AhA, curWaterFat);
                for (int k4 = 0; k4 < N; k4++) {
                    for (int k5 = 0; k5 < 2; k5++) { // 2 elements for water and fat currently
                        out(k1, k2, 0, 0, k4, k5, 0) = curWaterFat(k5, k4);
                    }
                }

            }
        }


        sqrt_inplace(&second_deriv);

//        fat_water_mixed_fitting(field_map,r2star_map,out,data,second_deriv, a,echoTimes,fieldStrength);



        out = *upsample<std::complex<float>,2>(&out);
        field_map = *upsample<float,2>(&field_map);
        r2star_map = *upsample<float,2>(&r2star_map);

//        fat_water_mixed_fitting(field_map, r2star_map, out, data_orig,second_deriv,
//                                a, echoTimes, fieldStrength);

        write_nd_array<float>(&field_map,"field_map.real");
        write_nd_array<float>(&r2star_map,"r2star_map.real");
        //Clean up as needed


        return out;
    }

}
