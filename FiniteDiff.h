#ifndef __FINITEDIFF_
#define __FINITEDIFF__

#include <petsc.h>
#include <petsc/private/dmdaimpl.h>
#include <iostream>
#include <math.h>
#include <TopOpt.h>
#include <Filter.h>
#include <LinearElasticity.h>


class FiniteDiff{
	
public:
	// Constructor
	FiniteDiff(TopOpt *opt);
	
	// Destructor
	~FiniteDiff();
	
	// Compute objective and constraints for the optimiation
	PetscErrorCode FiniteDiffCheck(TopOpt *opt); 
	
	
private:
	

	
};

#endif
