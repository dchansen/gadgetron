#include "CPUGriddingReconGadget.h"
#include "mri_core_grappa.h"
#include "vector_td_utilities.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "hoNFFT.h"
#include "hoNDArray.h"
#include "hoNDArray_elemwise.h"
#include "hoNDArray_math.h"
#include "hoCgSolver.h"
#include "ImageArraySendMixin.h"
#include <time.h>
#include <boost/range/algorithm/for_each.hpp>
#include "NonCartesianTools.h"

namespace Gadgetron{


	int CPUGriddingReconGadget::process_config(ACE_Message_Block *mb){
		ISMRMRD::IsmrmrdHeader h;
		deserialize(mb->rd_ptr(), h);
		auto matrixSize = h.encoding.front().encodedSpace.matrixSize;

		kernelWidth = kernelWidthProperty.value();
		oversamplingFactor = oversamplingFactorProperty.value();

		imageDims.push_back(matrixSize.x); 
		imageDims.push_back(matrixSize.y);
		
		imageDimsOs.push_back(matrixSize.x*oversamplingFactor);
		imageDimsOs.push_back(matrixSize.y*oversamplingFactor);
		this->initialize_encoding_space_limits(h);

		return GADGET_OK;
	}

	int CPUGriddingReconGadget::process(GadgetContainerMessage<IsmrmrdReconData> *m1){
		std::unique_ptr<GadgetronTimer> timer;
		if (perform_timing) {  timer = std::make_unique<GadgetronTimer>("CPUGridding");}
		IsmrmrdReconData *recon_bit_ = m1->getObjectPtr();

		for(size_t e = 0; e < recon_bit_->rbit_.size(); e++){
			IsmrmrdDataBuffered* buffer = &(recon_bit_->rbit_[e].data_);
			IsmrmrdImageArray imarray;
			
			size_t RO = buffer->data_.get_size(0);
			size_t E1 = buffer->data_.get_size(1);
			size_t E2 = buffer->data_.get_size(2);
			size_t CHA = buffer->data_.get_size(3);
			size_t N = buffer->data_.get_size(4);
			size_t S = buffer->data_.get_size(5);
			size_t SLC = buffer->data_.get_size(6);

//			imarray.data_.create(imageDims[0], imageDims[1], 1, 1, N, S, SLC);

			auto &trajectory = *buffer->trajectory_;
			auto trajDcw = separateDcwAndTraj(&trajectory);

			boost::shared_ptr<hoNDArray<float>> dcw = 
				boost::make_shared<hoNDArray<float>>(std::get<1>(trajDcw).get());
			boost::shared_ptr<hoNDArray<floatd2>> traj = 
				boost::make_shared<hoNDArray<floatd2>>(std::get<0>(trajDcw).get());
			
			std::vector<size_t> newOrder = {0, 1, 2, 4, 5, 6, 3};
			auto permuted = permute((hoNDArray<float_complext>*)&buffer->data_,&newOrder);
			hoNDArray<float_complext> data(*permuted);

			auto image = reconstruct(&data, traj.get(), dcw.get(), CHA);
			imarray.data_ = hoNDArray<std::complex<float>>(image->get_dimensions());
			memcpy(imarray.data_.get_data_ptr(),image->get_data_ptr(),image->get_number_of_bytes());

			NonCartesian::append_image_header(imarray,recon_bit_->rbit_[e], e);
			this->send_out_image_array(imarray, e, ((int)e + 1), GADGETRON_IMAGE_REGULAR);
		}

		m1->release();
		return GADGET_OK;
	}

	boost::shared_ptr<hoNDArray<float_complext>> CPUGriddingReconGadget::reconstruct(
		hoNDArray<float_complext> *data,
		hoNDArray<floatd2> *traj,
		hoNDArray<float> *dcw,
		size_t nCoils
	){

		hoNFFT_plan<float, 2> plan(
				from_std_vector<size_t, 2>(imageDims),
				oversamplingFactor,
				kernelWidth
		);

		hoNDArray<float_complext> result(imageDimsOs[0], imageDimsOs[1], data->get_number_of_elements()/traj->get_number_of_elements());
		plan.preprocess(*traj);
		plan.compute(*data, result, dcw, NFFT_comp_mode::BACKWARDS_NC2C);



//		write_nd_array(abs(&result).get(),"coils.reals");

		boost::for_each(result,[](auto & r){ r = norm(r);});
//		write_nd_array(abs(&result).get(),"squared.real");

		auto summed = sum(&result,2);
		sqrt_inplace(summed.get());
		write_nd_array(abs(summed.get()).get(),"summed.real");
		auto image_dims = from_std_vector<size_t ,2>(imageDims);
        auto image_dims_os = from_std_vector<size_t ,2>(imageDimsOs);

        summed = crop((image_dims_os-image_dims)/size_t(2),image_dims,summed.get());


		return summed;
	}	

	hoNDArray<float_complext> CPUGriddingReconGadget::reconstructChannel(
		hoNDArray<float_complext> *data,
		hoNDArray<floatd2> *traj,
		hoNDArray<float> *dcw
	){	
		if(!iterateProperty.value()){

			hoNFFT_plan<float, 2> plan(
				from_std_vector<size_t, 2>(imageDims),
				oversamplingFactor,
				kernelWidth
			);	
			hoNDArray<float_complext> result(imageDimsOs[0], imageDimsOs[1]);
			plan.preprocess(*traj);
			plan.compute(*data, result, dcw, NFFT_comp_mode::BACKWARDS_NC2C);

			return result;
		}else{
		    throw std::runtime_error("Iterative recon not implemented yet");
			// do iterative reconstruction
			//return boost::make_shared<hoNDArray<float_complext>>();
		}
	}

	std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>>
	CPUGriddingReconGadget::separateDcwAndTraj(
		hoNDArray<float> *dcwTraj
	){
		std::vector<size_t> dims = *dcwTraj->get_dimensions();
		std::vector<size_t> reducedDims(dims.begin()+1, dims.end());
		auto dcw = boost::make_shared<hoNDArray<float>>(reducedDims);
		auto traj = boost::make_shared<hoNDArray<floatd2>>(reducedDims);

		auto dcwPtr = dcw->get_data_ptr();
		auto trajPtr = traj->get_data_ptr();
		auto ptr = dcwTraj->get_data_ptr();
		for(unsigned int i = 0; i != dcwTraj->get_number_of_elements()/3; i++){
			trajPtr[i][0] = ptr[i*3];
			trajPtr[i][1] = ptr[i*3+1];
			dcwPtr[i] = ptr[i*3+2];
		}
		return std::make_tuple(traj, dcw);
	}

    CPUGriddingReconGadget::CPUGriddingReconGadget() {

    }

    CPUGriddingReconGadget::~CPUGriddingReconGadget() {

    }

    GADGET_FACTORY_DECLARE(CPUGriddingReconGadget);
}
