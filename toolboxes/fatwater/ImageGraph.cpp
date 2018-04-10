//
// Created by dch on 09/04/18.
//

#include "ImageGraph.h"


Gadgetron::vector_td<int,2> ImageGraph::index_to_offset[8] = {Gadgetron::vector_td<int,2>(-1,0),
                               Gadgetron::vector_td<int,2>(-1,-1),
                               Gadgetron::vector_td<int,2>(0,-1),
                               Gadgetron::vector_td<int,2>(1,0),
                               Gadgetron::vector_td<int,2>(-1,1),
                               Gadgetron::vector_td<int,2>(0,1),
                               Gadgetron::vector_td<int,2>(1,1)};

size_t ReverseEdgeMap::operator[](size_t e) const {
    return graph->reverse(e);
}