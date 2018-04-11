//
// Created by dch on 09/04/18.
//

#include "ImageGraph.h"


Gadgetron::vector_td<int,2> ImageGraph::index_to_offset[8] = {Gadgetron::vector_td<int,2>(-1,0),
                               Gadgetron::vector_td<int,2>(-1,1),
                               Gadgetron::vector_td<int,2>(0,1),
                               Gadgetron::vector_td<int,2>(1,1),
                               Gadgetron::vector_td<int,2>(1,0),
                               Gadgetron::vector_td<int,2>(1,-1),
                               Gadgetron::vector_td<int,2>(0,-1),
                               Gadgetron::vector_td<int,2>(-1,-1)};

size_t ReverseEdgeMap::operator[](size_t e) const {
    return graph->reverse(e);
}
size_t get(ReverseEdgeMap& m, size_t i){
    return m[i];
}


std::pair<ImageGraph::vertex_iterator, ImageGraph::vertex_iterator > vertices(const ImageGraph& g){

    return std::make_pair(g.vertex_begin(),g.vertex_end());

};




size_t num_vertices(const ImageGraph& g){
    return g.num_vertices();
}


size_t num_edges(const ImageGraph& g){
    return g.num_edges();
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


std::vector<float> &boost::get(boost::edge_capacity_t, ImageGraph &g){
    return g.edge_capacity_map;
}

std::vector<float> &boost::get(boost::edge_residual_capacity_t, ImageGraph &g){
    return g.edge_residual_capicty;
}

std::vector<boost::default_color_type> &boost::get(boost::vertex_color_t, ImageGraph &g){
    return g.color_map;
}

std::vector<float> &boost::get(boost::vertex_distance_t, ImageGraph &g){
    return g.vertex_distance;
}

const boost::identity_property_map &boost::get(boost::vertex_index_t, const ImageGraph &g){
    return g.vertex_index_map;
}

std::vector<size_t> &boost::get(boost::edge_reverse_t, ImageGraph &g){
    return g.reverse_edge_map;
}

std::vector<size_t> &boost::get(boost::vertex_predecessor_t, ImageGraph &g){
    return g.vertex_predecessor;

}
const std::vector<float> &boost::get(boost::edge_capacity_t, const ImageGraph &g){
    return g.edge_capacity_map;
}

const std::vector<float> &boost::get(boost::edge_residual_capacity_t, const ImageGraph &g){
    return g.edge_residual_capicty;
}

const std::vector<boost::default_color_type> &boost::get(boost::vertex_color_t, const ImageGraph &g){
    return g.color_map;
}

const std::vector<float> &boost::get(boost::vertex_distance_t, const ImageGraph &g){
    return g.vertex_distance;
}



const std::vector<size_t > &boost::get(boost::edge_reverse_t, const ImageGraph &g){
    return g.reverse_edge_map;
}
const std::vector<size_t> &boost::get(boost::vertex_predecessor_t, const ImageGraph &g){
    return g.vertex_predecessor;
}

