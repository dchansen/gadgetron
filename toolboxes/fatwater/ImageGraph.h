//
// Created by dch on 09/04/18.
//

#ifndef IMAGEGRAPH_H
#define IMAGEGRAPH_H
#pragma once
#
#include <boost/graph/graph_traits.hpp>

#include <boost/iterator/counting_iterator.hpp>
#include <vector>

template<class T> T& get(std::vector<T>& vec, size_t i){ return vec[i];}

template<class T, class R>
void put(std::vector<T> &vec, size_t i, R val) {
    vec[i] = val;
}

#include <boost/graph/properties.hpp>
#include "vector_td_utilities.h"



class ImageGraph {
public:

    constexpr static char edges_per_vertex = 10; // 8 per vertex + source & sink
    constexpr static char source_offset = 8;
    constexpr static char sink_offset = 9;
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


    ImageGraph(int x, int y){
        dims_ ={x,y};
        num_image_vertices_ = size_t(x)*size_t(y);
        num_vertices_ = num_image_vertices_ + 2;
        num_edges_ = (edges_per_vertex+2)*num_image_vertices_;
        source_vertex = num_image_vertices_;
        sink_vertex = source_vertex+1;

        setup_reverse_edge_map();
        reset();


    }


    void reset(){
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

    size_t num_vertices() const {
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
            size_t index_offset = e-normal_vertex_id*edges_per_vertex;
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
//    size_t out_degree(vertex_descriptor v) const;

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


        edge_descriptor  result;
        if (tv == source_vertex) {
            result = num_image_vertices_ * edges_per_vertex + sv;
        } else if (tv == sink_vertex){
            result =num_image_vertices_* (edges_per_vertex+1)+sv;

        } else if (sv == source_vertex){
            result = tv*edges_per_vertex+edges_per_vertex-2;
        } else if (sv == sink_vertex){
            result = tv*edges_per_vertex+edges_per_vertex-1;
        } else {

            auto sco = Gadgetron::idx_to_co(sv, dims_);
            auto tco = Gadgetron::idx_to_co(tv, dims_);

            auto diff = sco - tco;


            result =  tv * edges_per_vertex + get_edge_offset(diff);
        }

        auto result_target = target(result);
        auto result_source = source(result);
        assert(sv == result_target);
        assert(tv == result_source);
        return result;
    }


    size_t get_edge_offset(Gadgetron::vector_td<int,2> diff) const {
        size_t offset;

        if (diff[0] == 0) {
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 6;
            } else  {
                offset = 2;
            }
        } else if (diff[0] == -1 || diff[0] == dims_[0]-1){
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 7;
            } else if (diff[1] == 0){
                offset = 0;
            } else {
                offset = 1;
            }
        } else {
            if (diff[1] == -1 || diff[1] == dims_[1]-1) {
                offset = 5;
            } else if (diff[1] ==0){
                offset = 4;
            } else {
                offset = 3;
            }
        }
//        assert(diff == index_to_offset[offset]);
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





     size_t num_edges() const {
        return num_edges_;
    }


    std::pair<ImageGraph::edge_descriptor ,bool > add_edge(const ImageGraph::vertex_descriptor v1 , const ImageGraph::vertex_descriptor v2){




        edge_descriptor  result;

        if ((v1 == source_vertex || v1 == sink_vertex) && (v2 == source_vertex || v2 == sink_vertex))
            return std::make_pair(edge_descriptor(0),false);

        if (v1 == source_vertex) {
            result = num_image_vertices_ * edges_per_vertex + v2;
        } else if (v1 == sink_vertex){
            result =num_image_vertices_* (edges_per_vertex+1)+v2;

        } else if (v2 == source_vertex){
            result = v1*edges_per_vertex+edges_per_vertex-2;
        } else if (v2 == sink_vertex){
            result = v1*edges_per_vertex+edges_per_vertex-1;
        } else {

            auto co1 = Gadgetron::idx_to_co(v1, dims_);
            auto co2 = Gadgetron::idx_to_co(v2, dims_);

            auto diff = co2-co1;

            if (std::abs(diff[0]) > 1 || std::abs(diff[1]) > 1) return std::make_pair(edge_descriptor(0) ,false);

            result =  v1 * edges_per_vertex + get_edge_offset(diff);
        }

        return std::make_pair(result,true);



    }

    std::vector<float> edge_capacity_map;
    std::vector<float> edge_residual_capicty;
    std::vector<boost::default_color_type> color_map;
    std::vector<float> vertex_distance;
    std::vector<vertex_descriptor> vertex_predecessor;
    std::vector<edge_descriptor> reverse_edge_map;

    boost::identity_property_map vertex_index_map;
//    ReverseEdgeMap reverse_edge_map;

    vertex_descriptor source_vertex;
    vertex_descriptor sink_vertex;
private:

    void setup_reverse_edge_map(){
        reverse_edge_map = std::vector<edge_descriptor>(num_edges_);
        for (edge_descriptor edge = 0; edge < num_edges_; edge++){
            reverse_edge_map[edge] = reverse(edge);
        }

    }
    size_t num_vertices_;
    size_t num_image_vertices_;
    size_t num_edges_;

    Gadgetron::vector_td<int,2> dims_;




};

std::pair<ImageGraph::vertex_iterator, ImageGraph::vertex_iterator > vertices(const ImageGraph& g);


size_t num_vertices(const ImageGraph& g);

size_t num_edges(const ImageGraph& g);

ImageGraph::vertex_descriptor source(ImageGraph::edge_descriptor e, const ImageGraph& g);

ImageGraph::vertex_descriptor target(ImageGraph::edge_descriptor e, const ImageGraph& g);


std::pair<ImageGraph::edge_iterator, ImageGraph::edge_iterator> out_edges(ImageGraph::vertex_descriptor v, const ImageGraph& g);
size_t out_degree(ImageGraph::vertex_descriptor v, const ImageGraph& g);


std::pair<ImageGraph::edge_iterator, ImageGraph::edge_iterator> edges(const ImageGraph& g);



std::pair<ImageGraph::edge_descriptor ,bool> add_edge(ImageGraph::vertex_descriptor v1, ImageGraph::vertex_descriptor v2, ImageGraph& g);

namespace boost {

    float* get(boost::edge_capacity_t, ImageGraph &g);

    float* get(boost::edge_residual_capacity_t, ImageGraph &g);

    boost::default_color_type* get(boost::vertex_color_t, ImageGraph &g);

    float* get(boost::vertex_distance_t, ImageGraph &g);

    const identity_property_map &get(boost::vertex_index_t, const ImageGraph &g);

    size_t* get(boost::edge_reverse_t, ImageGraph &g);

    size_t* get(boost::vertex_predecessor_t, ImageGraph &g);
    const float* get(boost::edge_capacity_t, const ImageGraph &g);

    const float* get(boost::edge_residual_capacity_t, const ImageGraph &g);

    const boost::default_color_type* get(boost::vertex_color_t, const ImageGraph &g);

    const float* get(boost::vertex_distance_t, const ImageGraph &g);



    const size_t* get(boost::edge_reverse_t, const ImageGraph &g);
    const size_t* get(boost::vertex_predecessor_t, const ImageGraph &g);



}


namespace boost {



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


    template<> struct property_map<ImageGraph,edge_capacity_t>{ typedef float* type; typedef const float* const_type; };
    template<> struct property_map<ImageGraph,edge_residual_capacity_t>{ typedef float* type; typedef const float* const_type;};
    template<> struct property_map<ImageGraph,edge_reverse_t >{ typedef size_t* type; typedef const size_t* const_type;};

    template<> struct property_map<ImageGraph,vertex_color_t >{ typedef default_color_type* type; typedef const default_color_type* const_type;};
    template<> struct property_map<ImageGraph,vertex_distance_t >{ typedef float* type; typedef const float* const_type;};
    template<> struct property_map<ImageGraph,vertex_index_t >{ typedef identity_property_map type; typedef identity_property_map const_type;};
    template<> struct property_map<ImageGraph,vertex_predecessor_t >{ typedef size_t* type; typedef const size_t* const_type;};

}

#endif