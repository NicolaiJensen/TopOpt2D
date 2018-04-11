#ifndef __WRITEVTS_
#define __WRITEVTS__

#include <petsc.h>
#include <petsc/private/dmdaimpl.h>
#include <iostream>
#include <math.h>
#include <TopOpt.h>

class WriteVTS{
	
public:
	// Constructor
	WriteVTS(TopOpt *opt);
	
	// Destructor
	~WriteVTS();
	
	// Compute objective and constraints for the optimiation
	PetscErrorCode WritetoVTS(TopOpt *opt, PetscInt itr); 
	
	
private:
        std::string filenameI;
        char filenameChar[PETSC_MAX_PATH_LEN];
        PetscBool flg;

	
};

#endif
