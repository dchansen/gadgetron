//
// Created by dch on 09/04/18.
//
#pragma once

#include <boost/graph/graph_traits.hpp>

#include <boost/iterator/counting_iterator.hpp>
class ImageGraph;
template<class T> T& get(std::vector<T>& vec, size_t i){ return vec[i];}
class ReverseEdgeMap {
public:
    ReverseEdgeMap(ImageGraph* g) : graph(g){};

    size_t operator[](size_t e) const;

private:
    ImageGraph* graph;
};

size_t get(ReverseEdgeMap&, size_t);


template<class T, class R>
void put(std::vector<T> &vec, size_t i, R val) {
    vec[i] = val;
}

#include <boost/graph/properties.hpp>
#include "vector_td_utilities.h"



class ImageGraph {
public:

    constexpr static char edges_per_vertex = 11; // 8 per vertex + source & sink
    constexpr static char source_offset = 9;
    constexpr static char sink_offset = 10;
     static  Gadgetron::vector_td<int,2> index_to_offset[8];
    typedef size_t vertex_descriptor;
    typedef size_t vertices_size_type;

    typedef size_t edges_size_type;
    typedef size_t edge_descriptor;

    class traversal_category : public boost::vertex_list_graph_tag,
                               public boost::edge_list_graph_tag,
                               public boost::incidence_graph_tag    {};



    typedef boost::counting_iterator<size_t> vertex_iterator;
    typedef boost::counting_iterator<size_t> edge_iterator;


    ImageGraph(int x, int y): reverse_edge_map(this){
        dims_ ={x,y};
        num_image_vertices_ = size_t(x)*size_t(y);
        num_vertices_ = num_image_vertices_ + 2;
        num_edges_ = (edges_per_vertex+2)*num_image_vertices_;
        source_vertex = num_image_vertices_;
        sink_vertex = source_vertex+1;
        edge_capacity_map = std::vector<float>(num_edges_,0);
        edge_residual_capicty = std::vector<float>(num_edges_,0);
        color_map = std::vector<boost::default_color_type>(num_vertices_,boost::default_color_type::gray_color);
        vertex_distance = std::vector<float>(num_vertices_,0);
        vertex_predecessor = std::vector<vertex_descriptor>(num_vertices_, 0);


    }


    vertex_iterator vertex_begin() const {
        return vertex_iterator(0);
    }

    vertex_iterator vertex_end() const {
        return vertex_iterator(num_vertices_);
    }

    size_t get_num_vertices() const {
        return num_vertices_;
    }


    edge_iterator edge_begin() const{
        return edge_iterator(0);
    }

    edge_iterator edge_end() const {
        return edge_iterator(num_edges_);
    }


    vertex_descriptor source(edge_descriptor e) const {

        size_t normal_vertex_id = e/edges_per_vertex;
        if  (normal_vertex_id < num_image_vertices_){
            return normal_vertex_id;
        }

        size_t remainder = e - num_image_vertices_ * edges_per_vertex;
        if (remainder < num_image_vertices_){
            return source_vertex;
        } else{
            return sink_vertex;
        }
    }



    const vertex_descriptor target(edge_descriptor e) const {
        size_t normal_vertex_id = e/edges_per_vertex;
        if (normal_vertex_id < num_image_vertices_){
            size_t index_offset = e-normal_vertex_id;
            if (index_offset < edges_per_vertex-2) {
                auto offset = index_to_offset[index_offset];
                auto co = Gadgetron::idx_to_co(normal_vertex_id, dims_);
                auto co2 = (co + offset +dims_) % dims_;
                return Gadgetron::co_to_idx(co2,dims_);
            } else {
                if (index_offset == edges_per_vertex-2)
                    return source_vertex;
                return sink_vertex;
            }
        } else {
            return e%num_image_vertices_;
        }
    }


    std::pair<edge_iterator,edge_iterator> out_edges(vertex_descriptor v) const {
        if (v < num_image_vertices_) {
            return std::make_pair(edge_iterator(v * edges_per_vertex), edge_iterator((v + 1) * edges_per_vertex));
        } else {
            if (v == source_vertex) {
                edge_descriptor source_start = num_image_vertices_* edges_per_vertex;
                return std::make_pair(edge_iterator(source_start),edge_iterator(source_start+num_image_vertices_));
            } else {
                edge_descriptor source_start = num_image_vertices_* (edges_per_vertex + 1);
                return std::make_pair(edge_iterator(source_start),edge_iterator(source_start+num_image_vertices_));
            }
        }

    }

    size_t out_degree(vertex_descriptor v) const {
        if (v < num_image_vertices_) {
            return edges_per_vertex;
        } else {
            return num_image_vertices_;
        }
    }


    edge_descriptor reverse(edge_descriptor e ) const {

        vertex_descriptor sv = source(e);
        vertex_descriptor tv = target(e);

        if (tv == source_vertex) {
            return num_image_vertices_ * edges_per_vertex + sv;
        }
        if (tv == sink_vertex){
            return num_image_vertices_* (edges_per_vertex+1)+sv;

        }
        if (sv == source_vertex){
            return tv*edges_per_vertex+edges_per_vertex-2;
        }
        if (sv == sink_vertex){
            return tv*edges_per_vertex+edges_per_vertex-1;
        }

        auto sco = Gadgetron::idx_to_co(sv,dims_);
        auto tco = Gadgetron::idx_to_co(sv,dims_);

        auto diff = sco-tco;



        return tv*edges_per_vertex+get_edge_offset(diff);
    }


    size_t get_edge_offset(Gadgetron::vector_td<int,2> diff) const {
        size_t offset;

        if (diff[0] == 0) {
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 0;
            } else  {
                offset = 4;
            }
        } else if (diff[0] == -1 || diff[0] == dims_[0]-1){
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 1;
            } else if (diff[1] == 0){
                offset = 2;
            } else {
                offset = 3;
            }
        } else {
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 7;
            } else if (diff[1] ==0){
                offset = 6;
            } else {
                offset = 5;
            }
        }
        return offset;
    }

    edge_descriptor edge_to_source(vertex_descriptor v){
        return v*edges_per_vertex+source_offset;
    }

    edge_descriptor edge_to_sink(vertex_descriptor v){
        return v*edges_per_vertex+sink_offset;
    }


    edge_descriptor edge_from_source(vertex_descriptor v){
        return num_image_vertices_*edges_per_vertex+v;
    }

    edge_descriptor edge_from_sink(vertex_descriptor v){
        return num_image_vertices_*(edges_per_vertex+1)+v;
    }

    edge_descriptor edge_from_offset(vertex_descriptor v, Gadgetron::vector_td<int,2> offset){
        return v*edges_per_vertex+get_edge_offset(offset);
    }

//    std::pair<edge_descriptor, bool> edge(vertex_descriptor s, vertex_descriptor t){
//
//        //Check if vertices are valid
//        if (s >= num_vertices_ || t >= num_vertices_ ){
//            return std::make_pair(std::numeric_limits<size_t>::max(),false);
//        }
//        //Check if both source and target are in the sink
//        if (s > num_image_vertices_ && t > num_image_vertices_) {
//            return std::make_pair(std::numeric_limits<size_t>::max(),false);
//        }
//        //No self loops
//        if (s == t ) {
//            return std::make_pair(std::numeric_limits<size_t>::max(),false);
//        }
//
//
//        if (s == source_vertex){
//            edge_descriptor  e = num_image_vertices_*edges_per_vertex + t;
//            return std::make_pair(e,true);
//        }
//        if (s == sink_vertex){
//            edge_descriptor  e = num_image_vertices_*(edges_per_vertex+1) + t;
//            return std::make_pair(e,true);
//        }
//
//        if (t == source_vertex){
//            edge_descriptor e = edges_per_vertex*s+edges_per_vertex-2;
//            return std::make_pair(e,true);
//        }
//        if (t == sink_vertex){
//            edge_descriptor e = edges_per_vertex*s+edges_per_vertex-1;
//            return std::make_pair(e,true);
//        }
//
//        auto sco = Gadgetron::idx_to_co(s,dims_);
//        auto tco = Gadgetron::idx_to_co(t,dims_);
//
//        auto offset = tco-sco+dims_;
//        //Handle wrapping boundaries
//        if (offset[0] == (dims_[0]-1)) offset[0] = -1;
//        if (offset[0] == -(dims_[0]-1)) offset[0] = 1;
//        if (offset[1] == (dims_[1]-1)) offset[1] = -1;
//        if (offset[1] == -(dims_[1]-1)) offset[1] = -1;
//
//
//
//
//    };




     size_t get_num_edges() const {
        return num_edges_;
    }

    std::vector<float> edge_capacity_map;
    std::vector<float> edge_residual_capicty;
    std::vector<boost::default_color_type> color_map;
    std::vector<float> vertex_distance;
    std::vector<vertex_descriptor> vertex_predecessor;

    boost::identity_property_map vertex_index_map;
    ReverseEdgeMap reverse_edge_map;


private:
    size_t num_vertices_;
    size_t num_image_vertices_;
    size_t num_edges_;
    vertex_descriptor source_vertex;
    vertex_descriptor sink_vertex;
    Gadgetron::vector_td<int,2> dims_;




};


std::pair<ImageGraph::vertex_iterator, ImageGraph::vertex_iterator > vertices(const ImageGraph& g){

    return std::make_pair(g.vertex_begin(),g.vertex_end());

};


size_t num_vertices(const ImageGraph& g){
    return g.get_num_vertices();
}

size_t num_edges(const ImageGraph& g){
    return g.get_num_edges();
}

ImageGraph::vertex_descriptor source(ImageGraph::edge_descriptor e, const ImageGraph& g){
    return g.source(e);
}
ImageGraph::vertex_descriptor target(ImageGraph::edge_descriptor e, const ImageGraph& g){
    return g.target(e);
}

std::pair<ImageGraph::edge_iterator, ImageGraph::edge_iterator> out_edges(ImageGraph::vertex_descriptor v, const ImageGraph& g){
    return g.out_edges(v);
};

size_t out_degree(ImageGraph::vertex_descriptor v, const ImageGraph& g){
    return g.out_degree(v);
}

std::pair<ImageGraph::edge_iterator, ImageGraph::edge_iterator> edges(const ImageGraph& g){
    return std::make_pair(g.edge_begin(),g.edge_end());
};

namespace boost {

    std::vector<float> &get(boost::edge_capacity_t, ImageGraph &g) {
        return g.edge_capacity_map;
    }

    std::vector<float> &get(boost::edge_residual_capacity_t, ImageGraph &g) {
        return g.edge_residual_capicty;
    }

    std::vector<boost::default_color_type> &get(boost::vertex_color_t, ImageGraph &g) {
        return g.color_map;
    }

    std::vector<float> &get(boost::vertex_distance_t, ImageGraph &g) {
        return g.vertex_distance;
    }

    const identity_property_map &get(boost::vertex_index_t, const ImageGraph &g) {
        return g.vertex_index_map;
    }

    ReverseEdgeMap &get(boost::edge_reverse_t, ImageGraph &g) {
        return g.reverse_edge_map;
    }

    std::vector<size_t> &get(boost::vertex_predecessor_t, ImageGraph &g) {
        return g.vertex_predecessor;
    }
    const std::vector<float> &get(boost::edge_capacity_t, const ImageGraph &g) {
        return g.edge_capacity_map;
    }

    const std::vector<float> &get(boost::edge_residual_capacity_t, const ImageGraph &g) {
        return g.edge_residual_capicty;
    }

    const std::vector<boost::default_color_type> &get(boost::vertex_color_t, const ImageGraph &g) {
        return g.color_map;
    }

    const std::vector<float> &get(boost::vertex_distance_t, const ImageGraph &g) {
        return g.vertex_distance;
    }



    const ReverseEdgeMap &get(boost::edge_reverse_t, const ImageGraph &g) {
        return g.reverse_edge_map;
    }
    const std::vector<size_t> &get(boost::vertex_predecessor_t, const ImageGraph &g) {
        return g.vertex_predecessor;
    }






    float &get(std::vector<float>& vec, size_t i){
        return vec[i];
    }

    const float &get(const std::vector<float>& vec, size_t i){
        return vec[i];
    }

    template<class T>
    const T &get(const std::vector<T> &vec, size_t i) {
        return vec[i];
    }












}


namespace boost {

    template<class T> struct property_traits<std::vector<T>> {
        typedef T value_type;
        typedef size_t key_type;
        typedef T reference;
        typedef lvalue_property_map_tag category;
    };

    template<> struct property_traits<ReverseEdgeMap> {
        typedef ImageGraph::edge_descriptor value_type;
        typedef size_t key_type;
        typedef value_type& reference;
        typedef readable_property_map_tag category;
    };



    typedef ImageGraph* image_graph_ptr;
    typedef const ImageGraph* image_const_graph_ptr;



    template <> struct graph_traits<ImageGraph> {
        typedef ImageGraph::vertex_descriptor vertex_descriptor;
        typedef ImageGraph::edge_descriptor edge_descriptor;
        typedef ImageGraph::edge_iterator out_edge_iterator;
        typedef void in_edge_iterator;

        typedef ImageGraph::vertex_iterator vertex_iterator;
        typedef ImageGraph::edge_iterator edge_iterator;
        typedef size_t vertices_size_type;
        typedef size_t edges_size_type;
        typedef size_t degree_size_type;
        typedef directed_tag directed_category;
        typedef ImageGraph::traversal_category traversal_category;

        typedef disallow_parallel_edge_tag edge_parallel_category;
        static edge_descriptor null_vertex(){ return std::numeric_limits<size_t>::max();}



    };

    template <> struct graph_traits<image_const_graph_ptr> {
        typedef ImageGraph::vertex_descriptor vertex_descriptor;
        typedef ImageGraph::edge_descriptor edge_descriptor;
        typedef ImageGraph::edge_iterator out_edge_iterator;
        typedef void in_edge_iterator;

        typedef ImageGraph::vertex_iterator vertex_iterator;
        typedef ImageGraph::edge_iterator edge_iterator;
        typedef size_t vertices_size_type;
        typedef size_t edge_size_type;
        typedef size_t degree_size_type;
        typedef directed_tag directed_category;
        typedef ImageGraph::traversal_category traversal_category;


    };


    template<> struct property_map<ImageGraph,edge_capacity_t>{ typedef std::vector<float> type; typedef std::vector<float> const_type; };
    template<> struct property_map<ImageGraph,edge_residual_capacity_t>{ typedef std::vector<float> type; typedef std::vector<float> const_type;};
    template<> struct property_map<ImageGraph,edge_reverse_t >{ typedef ReverseEdgeMap type; typedef ReverseEdgeMap const_type;};

    template<> struct property_map<ImageGraph,vertex_color_t >{ typedef std::vector<default_color_type > type; typedef std::vector<default_color_type > const_type;};
    template<> struct property_map<ImageGraph,vertex_distance_t >{ typedef std::vector<float > type; typedef std::vector<float > const_type;};
    template<> struct property_map<ImageGraph,vertex_index_t >{ typedef identity_property_map type; typedef identity_property_map const_type;};
    template<> struct property_map<ImageGraph,vertex_predecessor_t >{ typedef std::vector<size_t> type; typedef std::vector<size_t> const_type;};

}