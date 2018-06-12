//
// Created by dch on 09/04/18.
//

#include "ImageGraph.h"

using namespace Gadgetron;

template<> Gadgetron::vector_td<int, 2> ImageGraph<2>::index_to_offset[4] = {Gadgetron::vector_td<int,2>(-1,0),
                               Gadgetron::vector_td<int,2>(1,0),
                               Gadgetron::vector_td<int,2>(0,-1),
                               Gadgetron::vector_td<int,2>(0,1)};


template<> Gadgetron::vector_td<int, 3> ImageGraph<3>::index_to_offset[6] = {Gadgetron::vector_td<int,3>(-1,0,0),
                                    Gadgetron::vector_td<int,3>(1,0,0),
                                    Gadgetron::vector_td<int,3>(0,-1,0),
                                    Gadgetron::vector_td<int,3>(0,1,0),
                                    Gadgetron::vector_td<int,3>(0,0,-1),
                                    Gadgetron::vector_td<int,3>(0,0,1)

                                    };



template<unsigned int D> std::pair<typename ImageGraph<D>::vertex_iterator, typename ImageGraph<D>::vertex_iterator > boost::vertices(const ImageGraph<D>& g){

    return std::make_pair(g.vertex_begin(),g.vertex_end());

};
template<unsigned int D>  std::pair<typename ImageGraph<D>::edge_descriptor ,bool> edge(typename ImageGraph<D>::vertex_descriptor v1, typename ImageGraph<D>::vertex_descriptor v2, const ImageGraph<D>& g){
    return g.edge(v1,v2);

};



template<unsigned int D> size_t num_vertices(const ImageGraph<D>& g){
    return g.num_vertices();
}


template<unsigned int D> size_t num_edges(const ImageGraph<D>& g){
    return g.num_edges();
}



template<unsigned int D> typename ImageGraph<D>::vertex_descriptor source(typename ImageGraph<D>::edge_descriptor e, const ImageGraph<D>& g){
    return g.source(e);
}



template<unsigned int D> typename ImageGraph<D>::vertex_descriptor target(typename ImageGraph<D>::edge_descriptor e, const ImageGraph<D>& g){
    return g.target(e);
}


template<unsigned int D> std::pair<typename ImageGraph<D>::edge_iterator, typename ImageGraph<D>::edge_iterator> out_edges(typename ImageGraph<D>::vertex_descriptor v, const ImageGraph<D>& g){
    return g.out_edges(v);
};




template<unsigned int D> size_t out_degree(typename ImageGraph<D>::vertex_descriptor v, const ImageGraph<D>& g){
    return g.out_degree(v);
}


template<unsigned int D> std::pair<typename ImageGraph<D>::edge_iterator, typename ImageGraph<D>::edge_iterator> edges(const ImageGraph<D>& g){
    return std::make_pair(g.edge_begin(),g.edge_end());
};


template<unsigned int D> float* boost::get(boost::edge_capacity_t, ImageGraph<D> &g){
    return g.edge_capacity_map.data();
}

template<unsigned int D> float* boost::get(boost::edge_residual_capacity_t, ImageGraph<D> &g){
    return g.edge_residual_capicty.data();
}

template<unsigned int D> boost::default_color_type* boost::get(boost::vertex_color_t, ImageGraph<D> &g){
    return g.color_map.data();
}

template<unsigned int D> float* boost::get(boost::vertex_distance_t, ImageGraph<D> &g){
    return g.vertex_distance.data();
}

template<unsigned int D> const boost::identity_property_map &boost::get(boost::vertex_index_t, const ImageGraph<D> &g){
    return g.vertex_index_map;
}

template<unsigned int D> size_t* boost::get(boost::edge_reverse_t, ImageGraph<D> &g){
    return g.reverse_edge_map.data();
}

template<unsigned int D> size_t* boost::get(boost::vertex_predecessor_t, ImageGraph<D> &g){
    return g.vertex_predecessor.data();

}
template<unsigned int D> const float* boost::get(boost::edge_capacity_t, const ImageGraph<D> &g){
    return g.edge_capacity_map.data();
}

template<unsigned int D> const float* boost::get(boost::edge_residual_capacity_t, const ImageGraph<D> &g){
    return g.edge_residual_capicty.data();
}

template<unsigned int D> const boost::default_color_type* boost::get(boost::vertex_color_t, const ImageGraph<D> &g){
    return g.color_map.data();
}

template<unsigned int D> const float* boost::get(boost::vertex_distance_t, const ImageGraph<D> &g){
    return g.vertex_distance.data();
}



template<unsigned int D> const size_t * boost::get(boost::edge_reverse_t, const ImageGraph<D> &g){
    return g.reverse_edge_map.data();
}
template<unsigned int D> const size_t* boost::get(boost::vertex_predecessor_t, const ImageGraph<D> &g){
    return g.vertex_predecessor.data();
}

