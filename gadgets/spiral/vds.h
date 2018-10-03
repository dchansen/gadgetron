#pragma once

#include "gadgetron_spiral_export.h"
#include "hoNDArray.h"
#include "vector_td.h"
namespace Gadgetron{

  void EXPORTGADGETS_SPIRAL 
  calc_vds(double slewmax,double gradmax,double Tgsample,double Tdsample,int Ninterleaves,
	   double* fov, int numfov,double krmax,
	   int ngmax, double** xgrad,double** ygrad,int* numgrad);
  
  void EXPORTGADGETS_SPIRAL 
  calc_traj(double* xgrad, double* ygrad, int ngrad, int Nints, double Tgsamp, double krmax,
	    double** x_trajectory, double** y_trajectory, double** weights);


  EXPORTGADGETS_SPIRAL
  hoNDArray<floatd2> create_rotations(const double *xgrad, const double *ygrad, int Ngrad, int Nints);

  EXPORTGADGETS_SPIRAL
  hoNDArray<floatd2> calculate_trajectories(const hoNDArray<floatd2>& gradients, float Tgsamp, float  krmax);

 EXPORTGADGETS_SPIRAL
  hoNDArray<float> calculate_weights(const hoNDArray<floatd2>& gradients, const hoNDArray<floatd2>& trajectories);
}
