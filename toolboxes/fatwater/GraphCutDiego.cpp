#include "GraphCutDiego.h"
#include <boost/config.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/timer/timer.hpp>

using namespace boost;

typedef float EdgeWeightType;

typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
typedef adjacency_list<vecS, vecS, directedS,
        property<vertex_name_t, std::string,
                property<vertex_index_t, long,
                        property<vertex_color_t, boost::default_color_type,
                                property<vertex_distance_t, float,
                                        property<vertex_predecessor_t, Traits::edge_descriptor> > > > >,
        property<edge_capacity_t, EdgeWeightType,
                property<edge_residual_capacity_t, EdgeWeightType,
                        property<edge_reverse_t, Traits::edge_descriptor> > > > Graph;
namespace {
// DH: Function to add a single edge to the graph within the graph cut algorithm
void
AddEdge(Traits::vertex_descriptor &v1, Traits::vertex_descriptor &v2, property_map<Graph, edge_reverse_t>::type &rev,
        const double capacity, Graph &g) {
    Traits::edge_descriptor e1 = add_edge(v1, v2, g).first;
    Traits::edge_descriptor e2 = add_edge(v2, v1, g).first;
    put(edge_capacity, g, e1, capacity);
    put(edge_capacity, g, e2, 0 * capacity);

    rev[e1] = e2;
    rev[e2] = e1;
}

}

namespace Gadgetron {
    hoNDArray <uint16_t> doGraphCut(const hoNDArray <uint16_t> &cur_ind, const hoNDArray <uint16_t> &next_ind,
                                    const hoNDArray<float> &residual, const hoNDArray<float> &lmap, int size_clique) {


        const auto X = cur_ind.get_size(0);
        const auto Y = cur_ind.get_size(1);
        const auto Z = cur_ind.get_size(2);
//	auto g = make_graph(cur_ind,next_ind,residual,lmap);
//	auto g = ImageGraph(X,Y);
//	auto &capacity_map = g.edge_capacity_map;
        Graph g; // DH: create a graph with 0 vertices

        property_map<Graph, edge_reverse_t>::type rev = get(edge_reverse, g);

        // DH: add a source and sink node, and store them in s and t, respectively. Also, add a node per pixel.
        Traits::vertex_descriptor s = add_vertex(g);
        hoNDArray <Traits::vertex_descriptor> v(X, Y, Z);
        for (int kx = 0; kx < X; kx++) {
            for (int ky = 0; ky < Y; ky++) {
                for (int kz = 0; kz < Z; kz++) {
                    v(kx, ky, kz) = add_vertex(g);
                }
            }
        }
        Traits::vertex_descriptor t = add_vertex(g);


//	// DH: Add edges to the graph (Start with residual related edges, then regularization related edges)
//	// DH: Keep track of the min and max edge values (initialized here with extreme values)
        int min_edge = 10000;
        int max_edge = -10000;

        float dist;
        float a, b, c, d;
        float curlmap;
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
                    AddEdge(s, v(kx, ky, kz), rev, val_sv, g);
                    AddEdge(v(kx, ky, kz), t, rev, val_vt, g);


// DH: Keep track of max and min edge weights
                    if (val_sv > max_edge)
                        max_edge = val_sv;
                    if (val_vt > max_edge)
                        max_edge = val_vt;

                    if (val_sv < min_edge)
                        min_edge = val_sv;
                    if (val_vt < min_edge)
                        min_edge = val_vt;


// DH: Now include the regularization edges based on Hernando et al, MRM 2010
                    for (int dx = -size_clique; dx <= size_clique; dx++) {
                        for (int dy = -size_clique; dy <= size_clique; dy++) {
                            for (int dz = -size_clique; dz <= size_clique; dz++) {

                                dist = pow(dx * dx + dy * dy + dz * dz, 0.5);

                                if (kx + dx >= 0 && kx + dx < X && ky + dy >= 0 && ky + dy < Y && kz + dz >= 0 &&
                                    kz + dz < Z && dist > 0) {

                                    curlmap = std::min(lmap(kx, ky, kz), lmap(kx + dx, ky + dy, kz + dz));

                                    a = curlmap / dist *
                                        pow(cur_ind(kx, ky, kz) - cur_ind(kx + dx, ky + dy, kz + dz), 2);
                                    b = curlmap / dist *
                                        pow(cur_ind(kx, ky, kz) - next_ind(kx + dx, ky + dy, kz + dz), 2);
                                    c = curlmap / dist *
                                        pow(next_ind(kx, ky, kz) - cur_ind(kx + dx, ky + dy, kz + dz), 2);
                                    d = curlmap / dist *
                                        pow(next_ind(kx, ky, kz) - next_ind(kx + dx, ky + dy, kz + dz), 2);


                                    AddEdge(v(kx, ky, kz), v(kx + dx, ky + dy, kz + dz), rev, (int) (b + c - a - d), g);
                                    AddEdge(s, v(kx, ky, kz), rev, (int) (std::max(float(0.0), c - a)), g);
                                    AddEdge(v(kx, ky, kz), t, rev, (int) (std::max(float(0.0), a - c)), g);
                                    AddEdge(s, v(kx + dx, ky + dy, kz + dz), rev, (int) (std::max(float(0.0), d - c)),
                                            g);
                                    AddEdge(v(kx + dx, ky + dy, kz + dz), t, rev, (int) (std::max(float(0.0), c - d)),
                                            g);

                                }

                            }
                        }
                    }

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
                    if (boost::get(colormap, ky+kx*Y+kz*X*Y+1) != boost::default_color_type::black_color) {
//                    if (colormap[1 + ky + kx * Y + kz * X * Y] != boost::default_color_type::black_color){
                        result(kx, ky, kz) = next_ind(kx, ky, kz);
                    }
                }
            }
        }

        return result;

    }
}


