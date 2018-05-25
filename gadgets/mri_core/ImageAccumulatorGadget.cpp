//
// Created by dchansen on 5/24/18.
//

#include <ismrmrd/xml.h>
#include <unordered_map>
#include <numeric>
#include "ImageAccumulatorGadget.h"


using namespace Gadgetron;


Gadgetron::ImageAccumulatorGadget::ImageAccumulatorGadget() {




}


static size_t image_dimension_from_string(std::string name){
    if (name == "X") return 0;
    if (name == "Y") return 1;
    if (name == "Z") return 2;
    if (name == "CHA") return 3;
    if (name == "N") return 4;
    if (name == "S") return 5;
    if (name == "LOC") return 6;

    throw std::runtime_error("Name " + name + " does not match an image dimension");

}

static size_t header_dimension_from_string(std::string name){
    if (name == "N") return 0;
    if (name == "S") return 1;
    if (name == "LOC") return 2;

    throw std::runtime_error("Name " + name + " does not match a header dimension);

}

template<class T> auto Gadgetron::ImageAccumulatorGadget::extract_value(T &val) {

    auto dimension = accumulate_dimension.value();
    if (dimension == "average") return val.average;
    if (dimension == "slice") return val.slice;
    if (dimension == "contrast") return val.contrast;
    if (dimension == "phase") return val.phase;
    if (dimension == "repetition") return val.repetition;
    if (dimension == "set") return val.set;

    throw std::runtime_error("Unknown dimension type " + dimension);
}





int Gadgetron::ImageAccumulatorGadget::process_config(ACE_Message_Block *mb) {

    ISMRMRD::IsmrmrdHeader h;
    ISMRMRD::deserialize(mb->rd_ptr(),h);

    auto limits_opt = extract_value(h.encoding[0].encodingLimits);
    if (!limits_opt) throw std::runtime_error("Encoding limits not set in data for dimension " + accumulate_dimension.value());
    ISMRMRD::Limit limits = limits_opt.get();



    required_values = std::vector<uint16_t>();
    for (auto val = limits.minimum; val <= limits.maximum; val++) required_values.push_back(val);




    return GADGET_OK;


}

bool Gadgetron::ImageAccumulatorGadget::same_size(std::vector<Gadgetron::IsmrmrdImageArray>& values){

    if (values.size() <= 1)
        return true;

    auto& im = values.front().data_;
    return std::all_of(values.begin()+1,values.end(),[&](auto & im2 ){return im.dimensions_equal(im2.data_);});

}



template<class T> hoNDArray<T> combine(std::vector<hoNDArray<T>> arrays) {

    std::vector<size_t> new_dimensions = *arrays.front().get_dimensions();
    new_dimensions.push_back(arrays.size());

    hoNDArray<T> result(new_dimensions);

    T* data_ptr = result.get_data_ptr();

    for (auto arr : arrays) {
        std::copy(arr.begin(),arr.end(),data_ptr);
        data_ptr += arr.get_number_of_elements();
    }

    return result;
}





Gadgetron::IsmrmrdImageArray
Gadgetron::ImageAccumulatorGadget::combine_images(std::vector<Gadgetron::IsmrmrdImageArray>& images) {

    if (!same_size(images)) throw std::runtime_error("Images do not have the same size");

    size_t combine_dimension = image_dimension_from_string(combine_along.value());


    std::vector<hoNDArray<std::com




    }







}

int Gadgetron::ImageAccumulatorGadget::process(Gadgetron::GadgetContainerMessage<Gadgetron::IsmrmrdImageArray> *m1) {


    auto& image_array = *m1->getObjectPtr();

    images.push_back(image_array);

    for (auto h : image_array.headers_)
        seen_values.emplace(extract_value(h));

    bool done = std::all_of(required_values.begin(),required_values.end(),
            [&](uint16_t val){return seen_values.count(val);});

    if (done){

    }
    return GADGET_OK;
}

