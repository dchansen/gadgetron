#pragma once


namespace Gadgetron {

    template<class ARRAY> NDArrayViewIterator {
    public:

    NDArrayViewIterator(ARRAY& array,size_t dim) :  base_array(array) {
        base_dims = std::vector<size_t>();
        base_dims.reserve(dim);
        for (size_t i = 0; i<= dim; i++)
            base_dims.push_back(array.get_size(i));

        base_size = std::accumulate(base_dims.begin(), base_dims.end(), size_t(1), std::multiplies<size_t>());
        current_view = 0;
        num_views = array.get_number_of_elements()/base_size();

    }


    NDArrayViewIterator &operator++() {
        current_view++;
        return (*this);
    }

    bool operator==(const NDArrayViewIterator &other) {
        return (&other.base_array == &base_array) && (other.base_size == base_size) &&
               (other.current_view == current_view);
    }

    ARRAY operator*() {
        return ARRAY(base_dims, base_array.get_data_ptr() + base_size * current_view);
    }


    static NDArrayViewIterator make_end(ARRAY& array, size_t dim){
        auto end = NDArrayViewIterator(std::forward(array),dim);
        end.current_view = end.num_views;
        return end;
    }

    private:
    std::vector <size_t> base_dims;
    size_t base_size;
    size_t current_view;
    size_t num_views;
    ARRAY &base_array;
    ARRAY array_view;

};

    template<class ARRAY> struct NDArrayViewRange{
        NDArrayViewRange(ARRAY& array, size_t dim){
            begin_iterator = NDArrayViewIteator<ARRAY>(std::forward(array),dim);
            end_iterator = NDArrayViewIteator<ARRAY>::make_end(std::forward(array),dim);
        }

        NDArrayViewIteator& begin(){
            return begin_iterator;
        }
        NDArrayViewIterator<ARRAY>& end(){
            return end_iterator;
        }
        NDArrayViewIterator begin_iterator;
        NDArrayViewIterator end_iterator;
    };

}