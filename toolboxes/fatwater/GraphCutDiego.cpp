#include "GraphCutDiego.h"
#include <boost/config.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
//#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/timer/timer.hpp>
#include "ImageGraph.h"
using namespace boost;

typedef float EdgeWeightType;

//typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
using Traits = ImageGraph;
//typedef adjacency_list<vecS, vecS, directedS,
//        property<vertex_name_t, std::string,
//                property<vertex_index_t, long,
//                        property<vertex_color_t, boost::default_color_type,
//                                property<vertex_distance_t, float,
//                                        property<vertex_predecessor_t, Traits::edge_descriptor> > > > >,
//        property<edge_capacity_t, EdgeWeightType,
//                property<edge_residual_capacity_t, EdgeWeightType,
//                        property<edge_reverse_t, Traits::edge_descriptor> > > > Graph;
using Graph = ImageGraph;
namespace {
// DH: Function to add a single edge to the graph within the graph cut algorithm
void
AddEdge(Traits::vertex_descriptor &v1, Traits::vertex_descriptor &v2, property_map<Graph, edge_reverse_t>::type &rev,
        const double capacity, Graph &g) {

    Traits::edge_descriptor e1 = add_edge(v1, v2, g).first;

    Traits::edge_descriptor e2 = add_edge(v2, v1, g).first;

    auto capacity_map = get(edge_capacity,g);


    capacity_map[e1] = capacity;

//    put(edge_capacity, g, e1, capacity);
//    put(edge_capacity, g, e2, 0 * capacity);

//    rev[e1] = e2;
//    rev[e2] = e1;
}

}

namespace Gadgetron {
    void AddRegularization(const hoNDArray<float> &lmap, int kx, int ky, int kz, int dx, int dy, int dz, float dist,
                       const hoNDArray <uint16_t> &cur_ind, const hoNDArray <uint16_t> &next_ind, Graph &g,
                       size_t* rev,
                       unsigned long &s, hoNDArray<unsigned long> &v, unsigned long &t) {
        float curlmap = std::min(lmap(kx, ky, kz), lmap(kx + dx, ky + dy, kz + dz));

        float a = curlmap / dist *
                                        pow(cur_ind(kx, ky, kz) - cur_ind(kx + dx, ky + dy, kz + dz), 2);
        float b = curlmap / dist *
                                        pow(cur_ind(kx, ky, kz) - next_ind(kx + dx, ky + dy, kz + dz), 2);
        float c = curlmap / dist *
                                        pow(next_ind(kx, ky, kz) - cur_ind(kx + dx, ky + dy, kz + dz), 2);
        float d = curlmap / dist *
                                        pow(next_ind(kx, ky, kz) - next_ind(kx + dx, ky + dy, kz + dz), 2);


        AddEdge(v(kx, ky, kz), v(kx + dx, ky + dy, kz + dz), rev,  round(b + c - a - d), g);
        AddEdge(s, v(kx, ky, kz), rev, round(std::max(float(0.0), c - a)), g);
        AddEdge(v(kx, ky, kz), t, rev, round(std::max(float(0.0), a - c)), g);
        AddEdge(s, v(kx + dx, ky + dy, kz + dz), rev,  round(std::max(float(0.0), d - c)),
                                            g);
        AddEdge(v(kx + dx, ky + dy, kz + dz), t, rev, round(std::max(float(0.0), c - d)),
                                            g);
    }

    hoNDArray <uint16_t> doGraphCut(const hoNDArray <uint16_t> &cur_ind, const hoNDArray <uint16_t> &next_ind,
                                    const hoNDArray<float> &residual, const hoNDArray<float> &lmap, int size_clique) {


        const auto X = cur_ind.get_size(0);
        const auto Y = cur_ind.get_size(1);
        const auto Z = cur_ind.get_size(2);
//	auto g = make_graph(cur_ind,next_ind,residual,lmap);
//	auto g = ImageGraph(X,Y);
//	auto &capacity_map = g.edge_capacity_map;
        Graph g(X,Y); // DH: create a graph with 0 vertices
        auto s = g.source_vertex;
        auto t= g.sink_vertex;

        property_map<Graph, edge_reverse_t>::type rev = get(edge_reverse, g);

        // DH: add a source and sink node, and store them in s and t, respectively. Also, add a node per pixel.
//        Traits::vertex_descriptor s = add_vertex(g);
        hoNDArray <Traits::vertex_descriptor> v(X, Y, Z);
        for (int kx = 0; kx < X; kx++) {
            for (int ky = 0; ky < Y; ky++) {
                for (int kz = 0; kz < Z; kz++) {
                    v(kx, ky, kz) = kx + ky * X + kz * X * Y;
                }
            }
        }
//        Traits::vertex_descriptor t = add_vertex(g);

        std::cout << "Why is this not running?" << std::endl;
//	// DH: Add edges to the graph (Start with residual related edges, then regularization related edges)
//	// DH: Keep track of the min and max edge values (initialized here with extreme values)
//        int min_edge = 10000;
//        int max_edge = -10000;

        for (int kx = 0; kx < X; kx++) {
            for (int ky = 0; ky < Y; ky++) {
                for (int kz = 0; kz < Z; kz++) {

                    size_t idx = kx + ky * X + kz * X * Y;

//float val_sv = std::max(float(0.0),residual(next_ind(kx,ky,kz),kx,ky,kz)-residual(cur_ind(kx,ky,kz),kx,ky,kz));
//float val_vt = std::max(float(0.0),residual(cur_ind(kx,ky,kz),kx,ky,kz)-residual(next_ind(kx,ky,kz),kx,ky,kz));
//AddEdge(s, v(kx,ky,kz), rev, (int)round(val_sv), g);
//AddEdge(v(kx,ky,kz), t, rev, (int)round(val_vt), g);

// DH: Collect current and potential next residual value
                    float resNext = residual(next_ind(kx, ky, kz), kx, ky, kz);
                    float resCur = residual(cur_ind(kx, ky, kz), kx, ky, kz);

// DH: Edge weights from source to current pixel node, and from current pixel node to sink
                    double val_sv = (double) (std::max(float(0.0), resNext - resCur));
                    double val_vt = (double) (std::max(float(0.0), resCur - resNext));



// DH: Add the edges
                    AddEdge(s, idx, rev, val_sv, g);
                    AddEdge(idx, t, rev, val_vt, g);


// DH: Keep track of max and min edge weights
//                    if (val_sv > max_edge)
//                        max_edge = val_sv;
//                    if (val_vt > max_edge)
//                        max_edge = val_vt;
//
//                    if (val_sv < min_edge)
//                        min_edge = val_sv;
//                    if (val_vt < min_edge)
//                        min_edge = val_vt;
                    if (kx > 0)
                        AddRegularization(lmap, kx, ky, kz, -1, 0, 0, 1, cur_ind, next_ind, g,
                                                      rev, s, v, t);
                    if (ky > 0)
                        AddRegularization(lmap, kx, ky, kz, 0, -1, 0, 1, cur_ind, next_ind, g,
                                      rev, s, v, t);

// DH: Now include the regularization edges based on Hernando et al, MRM 2010
//                    for (int dx = -size_clique; dx <= size_clique; dx++) {
//                        for (int dy = -size_clique; dy <= size_clique; dy++) {
//                            for (int dz = -size_clique; dz <= size_clique; dz++) {
//
//                                float dist = pow(dx * dx + dy * dy + dz * dz, 0.5);
//
//                                if (kx + dx >= 0 && kx + dx < X && ky + dy >= 0 && ky + dy < Y && kz + dz >= 0 &&
//                                    kz + dz < Z && dist > 0) {
//
//                                    AddRegularization(lmap, kx, ky, kz, dx, dy, dz, dist, cur_ind, next_ind, g,
//                                                      rev, s, v, t);
//
//
//                                }
//
//                            }
//                        }
//                    }

                }
            }
        }



//	// DH: Solve the min-cut/max-flow problem for the current graph in this iteration

        auto flow = boykov_kolmogorov_max_flow(g, s,
                                               t); // a list of sources will be returned in s, and a list of sinks will be returned in t



// std::cout << "Max flow is: " << flow << std::endl;

        auto capacity = get(edge_capacity, g);

        auto residual_capacity = get(edge_residual_capacity, g);

        auto colormap = get(vertex_color, g);

        hoNDArray <uint16_t> result = cur_ind;
// DH: Take the output of max-flow and apply the corresponding optimum jump at each pixel
        for (int kx = 0; kx < X; kx++) {
            for (int ky = 0; ky < Y; ky++) {
                for (int kz = 0; kz < Z; kz++) {
                    if (boost::get(colormap, kx+ky*X+kz*X*Y) != boost::default_color_type::black_color) {
//                    if (colormap[1 + ky + kx * Y + kz * X * Y] != boost::default_color_type::black_color){
                        result(kx, ky, kz) = next_ind(kx, ky, kz);
                    }
                }
            }
        }

        return result;

    }
}


