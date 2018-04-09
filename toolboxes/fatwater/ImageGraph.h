//
// Created by dch on 09/04/18.
//
#pragma once

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/graph/grid_graph.hpp>
#include "vector_td_utilities.h"


struct ImageEdge {

    size_t source;
    size_t target;

};

class ImageGraph {
public:

    constexpr char edges_per_vertex = 11; // 8 per vertex + source & sink
    const std::vector<Gadgetron::vector_td<int,2>> index_to_offset {Gadgetron::vector_td<int,2>(-1,0),
                                                                    Gadgetron::vector_td<int,2>(-1,-1),
                                                                    Gadgetron::vector_td<int,2>(0,-1),
                                                                    Gadgetron::vector_td<int,2>(1,0),
                                                                    Gadgetron::vector_td<int,2>(-1,1),
                                                                    Gadgetron::vector_td<int,2>(0,1),
                                                                    Gadgetron::vector_td<int,2>(1,1)};
    typedef size_t vertex_descriptor;
    typedef size_t vertices_size_type;

    typedef size_t edges_size_type;
    typedef size_t edge_descriptor;

    class traversal_category : public boost::vertex_list_graph_tag,
                               public boost::edge_list_graph_tag {};


    typedef boost::counting_iterator<size_t> vertex_iterator;
    typedef boost::counting_iterator<size_t> edge_iterator;


    vertex_iterator vertex_begin(){
        return vertex_iterator(0);
    }

    vertex_iterator vertex_end(){
        return vertex_iterator(num_vertices_);
    }

    size_t num_vertices(){
        return num_vertices_;
    }


    edge_iterator edge_begin(){
        return edge_iterator(0);
    }

    edge_iterator edge_end(){
        return edge_iterator(num_vertices_);
    }


    vertex_descriptor source(edge_descriptor e){

        size_t normal_vertex_id = e/edges_per_vertex;
        if  (normal_vertex_id < num_vertices_){
            return normal_vertex_id;
        }

        size_t remainder = e - num_vertices_ * edges_per_vertex;
        if (remainder < num_vertices_){
            return source_vertex;
        } else{
            return sink_vertex;
        }
    }



    vertex_descriptor target(edge_descriptor e){
        size_t normal_vertex_id = e/edges_per_vertex;
        if (normal_vertex_id < num_vertices_){
            size_t index_offset = e-normal_vertex_id;
            if (index_offset < edges_per_vertex-2) {
                auto offset = index_to_offset[index_offset];
                auto co = Gadgetron::idx_to_co(normal_vertex_id, dims_);
                auto co2 = (co + (offset + dims_)) % dims_;
                return Gadgetron::co_to_idx(co2);
            } else {
                if (index_offset == edges_per_vertex-2)
                    return source_vertex;
                return sink_vertex;
            }
        } else {
            return e%num_vertices_;
        }

    }

    size_t num_edges(){
        return num_edges_;
    }



private:

    size_t num_vertices_;
    size_t num_edges_;
    vertex_descriptor source_vertex;
    vertex_descriptor sink_vertex;
    Gadgetron::vector_td<size_t,2> dims_;



};


std::pair<ImageGraph::vertex_iterator, ImageGraph::vertex_iterator > vertices(ImageGraph& g){

    return std::pair(g.vertex_begin(),g.vertex_end());

};


size_t num_vertices(ImageGraph& g){
    return g.num_vertices();
}

size_t num_edges(ImageGraph& g){
    return g.num_edges();
}

ImageGraph::vertex_descriptor source(ImageGraph::edge_descriptor e, ImageGraph& g){
    return g.source(e);
}
ImageGraph::vertex_descriptor target(ImageGraph::edge_descriptor e, ImageGraph& g){
    return g.target(e);
}



