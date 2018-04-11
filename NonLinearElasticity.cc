#include <NonLinearElasticity.h>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

 Disclaimer:                                                              
 The authors reserves all rights but does not guaranty that the code is   
 free from errors. Furthermore, we shall not be liable in any event     
 caused by the use of the program.                                     
*/


NonLinearElasticity::NonLinearElasticity(TopOpt *opt){
        // Set pointers to null
        K = NULL;
        U = NULL;
        RHS = NULL;
        N = NULL;
        ksp = NULL;

        // Set up sitffness matrix, load vector and boundary conditions (Dirichlet) for the design problem
        SetUpLoadAndBC(opt);
}

NonLinearElasticity::~NonLinearElasticity(){
	// Deallocate
	VecDestroy(&U);
	VecDestroy(&RHS);
	VecDestroy(&N);
	MatDestroy(&K);
	KSPDestroy(&ksp);
}

PetscErrorCode NonLinearElasticity::SetUpLoadAndBC(TopOpt *opt){

	PetscErrorCode ierr;
	// Allocate matrix and the RHS and Solution vector and Dirichlet vector
	ierr = DMCreateMatrix(opt->da_nodes,&K); CHKERRQ(ierr);
	ierr = DMCreateGlobalVector(opt->da_nodes,&U); CHKERRQ(ierr);
	VecDuplicate(U,&RHS);
	VecDuplicate(U,&N);
	
        // Set the local stiffness matrix
	PetscScalar a = opt->dx; // x-side length
	PetscScalar b = opt->dy; // y-side length
	PetscScalar X[4] = {0.0, a, a, 0.0};
	PetscScalar Y[4] = {0.0, 0.0, b, b};

	// Compute the element stiffnes matrix - constant due to structured grid
        Quad4Isoparametric(X, Y, opt->nu,opt->AssType,KE);
        
	// Set the RHS and Dirichlet vector
	VecSet(N,1.0);
	VecSet(RHS,0.0);
        
	// Global coordinates and a pointer                                           
	Vec lcoor; // borrowed ref - do not destroy!
	PetscScalar *lcoorp;
        
	// Get local coordinates in local node numbering including ghosts       
	ierr = DMGetCoordinatesLocal(opt->da_nodes,&lcoor); CHKERRQ(ierr);
	VecGetArray(lcoor,&lcoorp); // Putting local coordinates into pointer
        
	// Get local dof number
	PetscInt nn;
	VecGetSize(lcoor,&nn); 
	// Set the values of boundaries and forces:
	for ( PetscInt i=0;i<nn;i++ ){
            // Set fixed-fixed bottom boundary
            if ( i % 2 == 0 && lcoorp[i+1] == opt->xc[2] ){
                VecSetValueLocal(N,i,0.0,INSERT_VALUES);
                VecSetValueLocal(N,i+1,0.0,INSERT_VALUES);
            }
            // Set compressive pressure on top boundary 
            if ( i % 2 == 0 && lcoorp[i+1] == opt->xc[3] ){
                VecSetValueLocal(RHS,i+1,opt->F0,INSERT_VALUES);
            }	
            
//             // Set fixed-fixed left boundary
//             if ( i % 2 == 0 && lcoorp[i]==opt->xc[0] ){
//                 VecSetValueLocal(N,i,0.0,INSERT_VALUES);
//                 VecSetValueLocal(N,i+1,0.0,INSERT_VALUES);
//             }
//             // Set compressive pressure on right boundary
//             if ( i % 2 == 0 && lcoorp[i]==opt->xc[1] ){
//                 VecSetValueLocal(RHS,i,opt->F0,INSERT_VALUES);
//             }	
	}
	
	// Assemble across processors
	VecAssemblyBegin(N);
	VecAssemblyBegin(RHS);
	VecAssemblyEnd(N);
	VecAssemblyEnd(RHS);
        
        // Clear usage of lcoorp
	VecRestoreArray(lcoor,&lcoorp);

	return ierr;
}

PetscErrorCode NonLinearElasticity::SolveState(TopOpt *opt){

	PetscErrorCode ierr;

	double t1,t2;
        
	t1 = MPI_Wtime();
	// Assemble the stiffness matrix
	ierr = AssembleStiffnessMatrix(opt); CHKERRQ(ierr);

	// Setup the solver
	if ( ksp==NULL ){
                ierr = SetUpSolver(opt); CHKERRQ(ierr);
	}
	else {
                ierr = KSPSetOperators(ksp,K,K); CHKERRQ(ierr);
                KSPSetUp(ksp); 
	}

	// Solve
	ierr = KSPSolve(ksp,RHS,U); CHKERRQ(ierr);

	// DEBUG
	// Get iteration number and residual from KSP
	PetscInt niter;
	PetscScalar rnorm;
	KSPGetIterationNumber(ksp,&niter);
	KSPGetResidualNorm(ksp,&rnorm); 
	PetscReal RHSnorm;
   	ierr = VecNorm(RHS,NORM_2,&RHSnorm); CHKERRQ(ierr);
	rnorm = rnorm/RHSnorm;
	t2 = MPI_Wtime();
        
	PetscPrintf(PETSC_COMM_WORLD,"State solver:  iter: %i, rerr.: %e, time: %f\n",niter,rnorm,t2-t1);

	return ierr;
}

PetscErrorCode NonLinearElasticity::ComputeObjectiveConstraints(TopOpt *opt) {

	// Error code
	PetscErrorCode ierr;

	// Solve state eqs 
	ierr = SolveState(opt); CHKERRQ(ierr); 

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_2D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);

	// Get pointer to the densities
	PetscScalar *xp;
	VecGetArray(opt->xPhys,&xp);

	// Get Solution
	Vec Uloc;
	DMCreateLocalVector(opt->da_nodes,&Uloc);
	DMGlobalToLocalBegin(opt->da_nodes,U,INSERT_VALUES,Uloc);
	DMGlobalToLocalEnd(opt->da_nodes,U,INSERT_VALUES,Uloc);

	// get pointer to local vector
	PetscScalar *up;
	VecGetArray(Uloc,&up);

	// Edof array
	PetscInt edof[8];

	opt->fx = 0.0;
	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<2;k++){
				edof[j*2+k] = 2*necon[i*nen+j]+k;
			}
		}
		// Use SIMP for stiffness interpolation
		PetscScalar uKu=0.0;
		for (PetscInt k=0;k<8;k++){
			for (PetscInt h=0;h<8;h++){	
				uKu += up[edof[k]]*KE[k*8+h]*up[edof[h]];
			}
		}
		// Add to objective
		opt->fx += (opt->Emin + PetscPowScalar(xp[i],opt->penal)*(opt->Emax - opt->Emin))*uKu;
	}

	// Allreduce fx[0]
	PetscScalar tmp=opt->fx;
	opt->fx=0.0;
	MPI_Allreduce(&tmp,&(opt->fx),1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);		

	// Compute volume constraint gx[0]
	opt->gx[0]=0;
	VecSum(opt->xPhys, &(opt->gx[0]));
	opt->gx[0]=opt->gx[0]/(((PetscScalar)opt->n))-opt->volfrac;


	VecRestoreArray(opt->xPhys,&xp);
	VecRestoreArray(Uloc,&up);
	VecDestroy(&Uloc);

	return(ierr);  

}

PetscErrorCode NonLinearElasticity::ComputeSensitivities(TopOpt *opt) {

	PetscErrorCode ierr;

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_2D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);
	
	// Get pointer to the densities
	PetscScalar *xp;
	VecGetArray(opt->xPhys,&xp);

	// Get Solution
	Vec Uloc;
	DMCreateLocalVector(opt->da_nodes,&Uloc);
	DMGlobalToLocalBegin(opt->da_nodes,U,INSERT_VALUES,Uloc);
	DMGlobalToLocalEnd(opt->da_nodes,U,INSERT_VALUES,Uloc);

	// get pointer to local vector
	PetscScalar *up;
	VecGetArray(Uloc,&up);

	// Get dfdx
	PetscScalar *df;
	VecGetArray(opt->dfdx,&df);

	// Edof array
	PetscInt edof[8];

	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<2;k++){
				edof[j*2+k] = 2*necon[i*nen+j]+k;
			}
		}
		// Use SIMP for stiffness interpolation
		PetscScalar uKu=0.0;
		for (PetscInt k=0;k<8;k++){
			for (PetscInt h=0;h<8;h++){	
				uKu += up[edof[k]]*KE[k*8+h]*up[edof[h]];
			}
		}
		// Set the Senstivity
		df[i]= -1.0 * opt->penal*PetscPowScalar(xp[i],opt->penal-1)*(opt->Emax - opt->Emin)*uKu;
	}
	// Compute volume constraint gx[0]
	VecSet(opt->dgdx[0],1.0/(((PetscScalar)opt->n)));

	VecRestoreArray(opt->xPhys,&xp);
	VecRestoreArray(Uloc,&up);
	VecRestoreArray(opt->dfdx,&df);
	VecDestroy(&Uloc);

	return(ierr);  

}

PetscErrorCode NonLinearElasticity::ComputeObjectiveConstraintsSensitivities(TopOpt *opt) {
	// Errorcode
	PetscErrorCode ierr;

	// Solve state eqs 
	ierr = SolveState(opt); CHKERRQ(ierr); 

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_2D(opt->da_nodes,&nel,&nen,&necon); CHKERRQ(ierr);
	//DMDAGetElements(da_nodes,&nel,&nen,&necon); // Still issue with elemtype change !

	// Get pointer to the densities
	PetscScalar *xp;
	VecGetArray(opt->xPhys,&xp);

	// Get Solution
	Vec Uloc;
	DMCreateLocalVector(opt->da_nodes,&Uloc);
	DMGlobalToLocalBegin(opt->da_nodes,U,INSERT_VALUES,Uloc);
	DMGlobalToLocalEnd(opt->da_nodes,U,INSERT_VALUES,Uloc);

	// get pointer to local vector
	PetscScalar *up;
	VecGetArray(Uloc,&up);

	// Get dfdx
	PetscScalar *df;
	VecGetArray(opt->dfdx,&df);

	// Edof array
	PetscInt edof[8];

	opt->fx = 0.0;
	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<2;k++){
				edof[j*2+k] = 2*necon[i*nen+j]+k;
			}
		}
		// Use SIMP for stiffness interpolation
		PetscScalar uKu=0.0;
		for (PetscInt k=0;k<8;k++){
			for (PetscInt h=0;h<8;h++){	
				uKu += up[edof[k]]*KE[k*8+h]*up[edof[h]];
			}
		}
		// Add to objective
		opt->fx += (opt->Emin + PetscPowScalar(xp[i],opt->penal)*(opt->Emax - opt->Emin))*uKu;
		// Set the Senstivity
		df[i]= -1.0 * opt->penal*PetscPowScalar(xp[i],opt->penal-1)*(opt->Emax - opt->Emin)*uKu;
	}

	// Allreduce fx[0]
	PetscScalar tmp=opt->fx;
	opt->fx=0.0;
	MPI_Allreduce(&tmp,&(opt->fx),1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);		

	// Compute volume constraint gx[0]
	opt->gx[0]=0;
	VecSum(opt->xPhys, &(opt->gx[0]));
	opt->gx[0]=opt->gx[0]/(((PetscScalar)opt->n))-opt->volfrac;
	VecSet(opt->dgdx[0],1.0/(((PetscScalar)opt->n)));

	VecRestoreArray(opt->xPhys,&xp);
	VecRestoreArray(Uloc,&up);
	VecRestoreArray(opt->dfdx,&df);
	VecDestroy(&Uloc);

	return(ierr);  

}


PetscErrorCode NonLinearElasticity::WriteRestartFiles(){
  
	PetscErrorCode ierr = 0;
	
	// Only dump data if correct allocater has been used
	if (!restart){
		return -1;
	}

	// Choose previous set of restart files
	if (flip){ flip = PETSC_FALSE; 	}	
	else {     flip = PETSC_TRUE; 	}

	// Open viewers for writing
	PetscViewer view; // vectors
	if (!flip){
		PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename00.c_str(),FILE_MODE_WRITE,&view);
	}
	else if (flip){
		PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename01.c_str(),FILE_MODE_WRITE,&view);
	}
	
	// Write vectors
	VecView(U,view);	
	
	// Clean up
	PetscViewerDestroy(&view);
	
	return ierr;
  
}


//##################################################################
//##################################################################
//##################################################################
// ######################## PRIVATE ################################
//##################################################################
//##################################################################

PetscErrorCode NonLinearElasticity::AssembleStiffnessMatrix(TopOpt *opt){

	PetscErrorCode ierr;

	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	ierr = DMDAGetElements_2D(opt->da_nodes,&nel,&nen,&necon);
	CHKERRQ(ierr);

	// Get pointer to the densities
	PetscScalar *xp;
	VecGetArray(opt->xPhys,&xp);

	// Zero the matrix
	MatZeroEntries(K);	

	// Edof array
	PetscInt edof[8];
	PetscScalar ke[8*8];

	// Loop over elements
	for (PetscInt i=0;i<nel;i++){
		// loop over element nodes
		for (PetscInt j=0;j<nen;j++){
			// Get local dofs
			for (PetscInt k=0;k<2;k++){
				edof[j*2+k] = 2*necon[i*nen+j]+k;
			}
		} 
		// Use SIMP for stiffness interpolation
		PetscScalar dens = opt->Emin + PetscPowScalar(xp[i],opt->penal)*(opt->Emax-opt->Emin);
		for (PetscInt k=0;k<8*8;k++){
			ke[k]=KE[k]*dens;
		}
		// Add values to the sparse matrix
		ierr = MatSetValuesLocal(K,8,edof,8,edof,ke,ADD_VALUES);
		CHKERRQ(ierr);
	}
	MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

	// Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
	// 1.: K = N'*K*N
	MatDiagonalScale(K,N,N);
	// 2. Add ones, i.e. K = K + NI, NI = I - N
	Vec NI;
	VecDuplicate(N,&NI);
	VecSet(NI,1.0);
	VecAXPY(NI,-1.0,N);
	MatDiagonalSet(K,NI,ADD_VALUES);

	// Zero out possible loads in the RHS that coincide
	// with Dirichlet conditions
	VecPointwiseMult(RHS,RHS,N);

	VecDestroy(&NI);
	VecRestoreArray(opt->xPhys,&xp);
	DMDARestoreElements(opt->da_nodes,&nel,&nen,&necon);

	return ierr;
}

PetscErrorCode NonLinearElasticity::SetUpSolver(TopOpt *opt){

	PetscErrorCode ierr;
	
	// CHECK FOR RESTART POINT
	restart = PETSC_TRUE;
	flip = PETSC_TRUE;  
	PetscBool flg, onlyDesign;
	onlyDesign = PETSC_FALSE;
	char filenameChar[PETSC_MAX_PATH_LEN];
	PetscOptionsGetBool(NULL,NULL,"-restart",&restart,&flg);
	PetscOptionsGetBool(NULL,NULL,"-onlyLoadDesign",&onlyDesign,&flg); // DONT READ DESIGN IF THIS IS TRUE
	
	// READ THE RESTART FILE INTO THE SOLUTION VECTOR(S)
	if (restart){
	    // THE FILES FOR WRITING RESTARTS
	    std::string filenameWorkdir = "./";
	    PetscOptionsGetString(NULL,NULL,"-workdir",filenameChar,sizeof(filenameChar),&flg);
	    if (flg){
		    filenameWorkdir = "";
		    filenameWorkdir.append(filenameChar);
	    }
	    filename00 = filenameWorkdir;
	    filename01 = filenameWorkdir;
	    filename00.append("/RestartSol00.dat");
	    filename01.append("/RestartSol01.dat");
	    
	    // CHECK FOR SOLUTION AND READ TO STATE VECTOR(s)
	    if (!onlyDesign){
		  // Where to read the restart point from
		  std::string restartFileVec = ""; // NO RESTART FILE !!!!!
		  // GET FILENAME
		  PetscOptionsGetString(NULL,NULL,"-restartFileVecSol",filenameChar,sizeof(filenameChar),&flg);
		  if (flg) {
		    restartFileVec.append(filenameChar);
		  }
		
		  // PRINT TO SCREEN
		  PetscPrintf(PETSC_COMM_WORLD,"# Restarting with solution (State Vector) from (-restartFileVecSol): %s \n",restartFileVec.c_str());
		  
		  // Check if files exist:
		  PetscBool vecFile = fexists(restartFileVec);
		  if (!vecFile) { PetscPrintf(PETSC_COMM_WORLD,"File: %s NOT FOUND \n",restartFileVec.c_str()); }
		  
		  // READ
		  if (vecFile){
		      PetscViewer view;
		      // Open the data files 
		      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,restartFileVec.c_str(),FILE_MODE_READ,&view);
		      
		      VecLoad(U,view);
		      
		      PetscViewerDestroy(&view);
		  }
	    }
	}
		
	PC pc;

	// The fine grid Krylov method
	KSPCreate(PETSC_COMM_WORLD,&(ksp));

	// SET THE DEFAULT SOLVER PARAMETERS
	// The fine grid solver settings
	PetscScalar rtol = 1.0e-5;
	PetscScalar atol = 1.0e-50;
	PetscScalar dtol = 1.0e3;
	PetscInt restart = 100;
	PetscInt maxitsGlobal = 200;

	// Coarsegrid solver
	PetscScalar coarse_rtol = 1.0e-8;
	PetscScalar coarse_atol = 1.0e-50;
	PetscScalar coarse_dtol = 1e3;
	PetscInt coarse_maxits = 30;
	PetscInt coarse_restart = 30;

	// Number of smoothening iterations per up/down smooth_sweeps
	PetscInt smooth_sweeps = 4;

	// Set up the solver
	ierr = KSPSetType(ksp,KSPFGMRES); // KSPCG, KSPGMRES
	CHKERRQ(ierr);

	ierr = KSPGMRESSetRestart(ksp,restart);
	CHKERRQ(ierr);

	ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxitsGlobal);
	CHKERRQ(ierr);

	ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
	CHKERRQ(ierr);

	ierr = KSPSetOperators(ksp,K,K);
	CHKERRQ(ierr);

	// The preconditinoer
	KSPGetPC(ksp,&pc);
	// Make PCMG the default solver
	PCSetType(pc,PCMG);

	// Set solver from options
	KSPSetFromOptions(ksp);

	// Get the prec again - check if it has changed
	KSPGetPC(ksp,&pc);

	// Flag for pcmg pc
	PetscBool pcmg_flag = PETSC_TRUE;
	PetscObjectTypeCompare((PetscObject)pc,PCMG,&pcmg_flag);

	// Only if PCMG is used
	if (pcmg_flag){ 

		// DMs for grid hierachy
		DM  *da_list,*daclist;
		Mat R;

		PetscMalloc(sizeof(DM)*opt->nlvls,&da_list);
		for (PetscInt k=0; k<opt->nlvls; k++) da_list[k] = NULL;
		PetscMalloc(sizeof(DM)*opt->nlvls,&daclist);
		for (PetscInt k=0; k<opt->nlvls; k++) daclist[k] = NULL;

		// Set 0 to the finest level
		daclist[0] = opt->da_nodes;

		// Coordinates
		PetscReal xmin=opt->xc[0], xmax=opt->xc[1], ymin=opt->xc[2], ymax=opt->xc[3];

		// Set up the coarse meshes
		DMCoarsenHierarchy(opt->da_nodes,opt->nlvls-1,&daclist[1]);
		for (PetscInt k=0; k<opt->nlvls; k++) {
			// NOTE: finest grid is nlevels - 1: PCMG MUST USE THIS ORDER ??? 
			da_list[k] = daclist[opt->nlvls-1-k];
			// THIS SHOULD NOT BE NECESSARY
			DMDASetUniformCoordinates(da_list[k],xmin,xmax,ymin,ymax,0.0,0.0);
		}

		// the PCMG specific options
		PCMGSetLevels(pc,opt->nlvls,NULL);
		PCMGSetType(pc,PC_MG_MULTIPLICATIVE); // Default
		ierr = PCMGSetCycleType(pc,PC_MG_CYCLE_V); CHKERRQ(ierr);
		PCMGSetGalerkin(pc,PETSC_TRUE);
		for (PetscInt k=1; k<opt->nlvls; k++) {
			DMCreateInterpolation(da_list[k-1],da_list[k],&R,NULL);
			PCMGSetInterpolation(pc,k,R);
			MatDestroy(&R);
		}

		// tidy up 
		for (PetscInt k=1; k<opt->nlvls; k++) { // DO NOT DESTROY LEVEL 0
			DMDestroy(&daclist[k]);
		}
		PetscFree(da_list);
		PetscFree(daclist);

		// AVOID THE DEFAULT FOR THE MG PART
		{ 
			// SET the coarse grid solver: 
			// i.e. get a pointer to the ksp and change its settings
			KSP cksp;
			PCMGGetCoarseSolve(pc,&cksp);
			// The solver
			ierr = KSPSetType(cksp,KSPGMRES); // KSPCG, KSPFGMRES

			ierr = KSPGMRESSetRestart(cksp,coarse_restart);

			ierr = KSPSetTolerances(cksp,coarse_rtol,coarse_atol,coarse_dtol,coarse_maxits);
			// The preconditioner
			PC cpc;
			KSPGetPC(cksp,&cpc);
			PCSetType(cpc,PCSOR); // PCSOR, PCSPAI (NEEDS TO BE COMPILED), PCJACOBI     

			// Set smoothers on all levels (except for coarse grid):
			for (PetscInt k=1;k<opt->nlvls;k++){
				KSP dksp;
				PCMGGetSmoother(pc,k,&dksp);
				PC dpc;
				KSPGetPC(dksp,&dpc);
				ierr = KSPSetType(dksp,KSPGMRES); // KSPCG, KSPGMRES, KSPCHEBYSHEV (VERY GOOD FOR SPD)

				ierr = KSPGMRESSetRestart(dksp,smooth_sweeps);
				ierr = KSPSetTolerances(dksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,smooth_sweeps); // NOTE in the above maxitr=restart;
				PCSetType(dpc,PCSOR);// PCJACOBI, PCSOR for KSPCHEBYSHEV very good   
			}
		}
	}

	// Write check to screen:
	// Check the overall Krylov solver
	KSPType ksptype;
	KSPGetType(ksp,&ksptype);
	PCType pctype;
	PCGetType(pc,&pctype);
	PetscInt mmax;
	KSPGetTolerances(ksp,NULL,NULL,NULL,&mmax);
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	PetscPrintf(PETSC_COMM_WORLD,"################# Linear solver settings #####################\n");
	PetscPrintf(PETSC_COMM_WORLD,"# Main solver: %s, prec.: %s, maxiter.: %i \n",ksptype,pctype,mmax);

	// Only if pcmg is used
	if (pcmg_flag){
		// Check the smoothers and coarse grid solver:
		for (PetscInt k=0;k<opt->nlvls;k++){
			KSP dksp;
			PC dpc;
			KSPType dksptype;
			PCMGGetSmoother(pc,k,&dksp);
			KSPGetType(dksp,&dksptype);
			KSPGetPC(dksp,&dpc);
			PCType dpctype;
			PCGetType(dpc,&dpctype);
			PetscInt mmax;
			KSPGetTolerances(dksp,NULL,NULL,NULL,&mmax);
			PetscPrintf(PETSC_COMM_WORLD,"# Level %i smoother: %s, prec.: %s, sweep: %i \n",k,dksptype,dpctype,mmax);
		}
	}
	PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");


	return(ierr);
}
PetscErrorCode NonLinearElasticity::DMDAGetElements_2D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]) {
	PetscErrorCode ierr;
	
	DM_DA          *da = (DM_DA*)dm->data;
	PetscInt       i,xs,xe,Xs,Xe;
	PetscInt       j,ys,ye,Ys,Ye;
	
	PetscInt       cnt=0, cell[4], ns=1, nn=4;
	PetscInt       c;
	
	if (!da->e) {
		if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=4;}
		ierr = DMDAGetCorners(dm,&xs,&ys,NULL,&xe,&ye,NULL);
		CHKERRQ(ierr);
		ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,NULL,&Xe,&Ye,NULL);
		CHKERRQ(ierr);
		xe    += xs; Xe += Xs; if (xs != Xs) xs -= 1;
		ye    += ys; Ye += Ys; if (ys != Ys) ys -= 1;
		
		da->ne = ns*(xe - xs - 1)*(ye - ys - 1);
		PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);
		
		for (j=ys; j<ye-1; j++) {
			for (i=xs; i<xe-1; i++) {
				cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) ;
				cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs);
				cell[2] = (i-Xs+1) + (j-Ys+1)*(Xe-Xs) ;
				cell[3] = (i-Xs  ) + (j-Ys+1)*(Xe-Xs) ;
				if (da->elementtype == DMDA_ELEMENT_Q1) {
					for (c=0; c<ns*nn; c++) da->e[cnt++] = cell[c];
				}
			}
		}
		
	}
	*nel = da->ne;
	*nen = nn;
	*e   = da->e;
	return(0);
}

PetscInt NonLinearElasticity::Quad4Isoparametric(PetscScalar *d, PetscScalar *X, PetscScalar *Y, PetscScalar nu, PetscInt AssType, PetscScalar *ke){
    	// Errorcode
	PetscErrorCode ierr;
        
	// 
	PetscScalar lambda = nu/((1.0+nu)*(1.0-2.0*nu));
	PetscScalar mu = 1.0/(2.0*(1.0+nu));
	// Constitutive matrix
	PetscScalar C[3][3];
	memset(C, 0, sizeof(C[0][0])*3*3);
	if (AssType>0){
		C[0][0]=1.0/(1.0-nu*nu);
		C[1][1]=1.0/(1.0-nu*nu);
		C[0][1]=nu/(1.0-nu*nu);
		C[1][0]=nu/(1.0-nu*nu);
		C[2][2]=1.0/2.0/(1.0+nu);
	}
	else{
		C[0][0]=lambda+2.0*mu;
		C[1][1]=lambda+2.0*mu;
		C[0][1]=lambda;
		C[1][0]=lambda;
		C[2][2]=mu;
	}
	
	// Gauss points (GP) and weigths
	// Two Gauss points in all directions (total of eight)
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626}; 
	// Corresponding weights
	PetscScalar W[2] = {1.0, 1.0};
	//
	PetscScalar dNdxi[4]; PetscScalar dNdeta[4]; 
        PetscScalar J[2][2];
	PetscScalar invJ[2][2];
	PetscScalar B0[3][8]; // Note: Small enough to be allocated on stack
	PetscScalar BL[3][8]; // Note: Small enough to be allocated on stack
        PetscScalar B[3][8]; // Note: Small enough to be allocated on stack
	PetscScalar G[4][8]; // Note: Small enough to be allocated on stack
	PetscScalar A[3][4]; // Note: Small enough to be allocated on stack
	PetscScalar sigma[3]; // Note: Small enough to be allocated on stack
	PetscScalar Theta[4]; // Note: Small enough to be allocated on stack
	memset(ke, 0, sizeof(ke[0])*8*8);
	
	// Perform the numerical integration
	for (PetscInt ii=0; ii<2; ii++){
		for (PetscInt jj=0; jj<2; jj++){
			// Integration points
			PetscScalar xi = GP[ii]; PetscScalar eta = GP[jj]; 
                        // Differentiated shape functions
                        DifferentiatedShapeFunctions(xi, eta, dNdxi, dNdeta);
                        // Jacobian
                        J[0][0] = Dot(dNdxi,X,4); J[0][1] = Dot(dNdxi,Y,4); 
                        J[1][0] = Dot(dNdeta,X,4); J[1][1] = Dot(dNdeta,Y,4); 
                        // Inverse and determinant
                        PetscScalar detJ = Inverse2M(J, invJ);
                        // Weight factor at this point
                        PetscScalar weight = W[ii]*W[jj]*detJ;
                        // Strain-displacement matrix
			memset(B0, 0, sizeof(B[0][0])*3*8); // zero out
			memset(BL, 0, sizeof(B[0][0])*3*8); // zero out
                        memset(B, 0, sizeof(B[0][0])*3*8); // zero out
                        // G and A matrix
                        memset(G, 0, sizeof(G[0][0])*4*8); // zero out
                        memset(A, 0, sizeof(A[0][0])*3*4); // zero out
                        // Theta and Sigma
                        memset(Theta, 0, sizeof(Theta[0])*4); // zero out
                        memset(sigma, 0, sizeof(sigma[0])*3); // zero out
                        
			for (PetscInt i=0; i<4; i++){
				B0[0][2*i]   = invJ[0][0]*dNdxi[i]+invJ[0][1]*dNdeta[i];
				B0[1][2*i+1] = invJ[1][0]*dNdxi[i]+invJ[1][1]*dNdeta[i];
				B0[2][2*i]   = invJ[1][0]*dNdxi[i]+invJ[1][1]*dNdeta[i];
				B0[2][2*i+1] = invJ[0][0]*dNdxi[i]+invJ[0][1]*dNdeta[i];
			}
			
			for (PetscInt i=0; i<4; i++){
				G[0][2*i]   = invJ[0][0]*dNdxi[i]+invJ[0][1]*dNdeta[i];
				G[1][2*i+1] = invJ[0][0]*dNdxi[i]+invJ[0][1]*dNdeta[i];
				G[2][2*i]   = invJ[1][0]*dNdxi[i]+invJ[1][1]*dNdeta[i];
				G[3][2*i+1] = invJ[1][0]*dNdxi[i]+invJ[1][1]*dNdeta[i];
			}
			// Calculating Theta
                        ierr = MatMult(G,d,Theta);
                        
                        // Set A
                        A[0][0] = Theta[0];
                        A[0][1] = Theta[1];
                        A[0][2] = 0;
                        A[0][3] = 0;
                        A[1][0] = 0;
                        A[1][1] = 0;
                        A[1][2] = Theta[2];
                        A[1][3] = Theta[3];
                        A[2][0] = Theta[2];
                        A[2][1] = Theta[3];
                        A[2][2] = Theta[0];
                        A[2][3] = Theta[1];
                        
			// Calculate BL
                        ierr = MatMatMult(A,G,MAT_REUSE_MATRIX,PETSC_DEFAULT,*BL);
                        
                        // Calculate B
                        for (PetscInt i=0; i<3; i++){
                            for (PetscInt j=0; j<8; i++){   
                                B[i][j] = B0[i][j] + BL[i][j];
                            }
                        }
                        // Calculate the epsilons (sigmas)
                        ierr = MatMult(B0+0.5*BL,d,sigma);
                        
			// Finally, add to the element matrix
			for (PetscInt i=0; i<8; i++){
				for (PetscInt j=0; j<8; j++){
					for (PetscInt k=0; k<3; k++){
						for (PetscInt l=0; l<3; l++){	
							ke[j+8*i] = ke[j+8*i] + weight*(B[k][i] * C[k][l] * B[l][j]);
							
						}
					}
				}
			}
		}
	}
	
	return(ierr);
}

PetscScalar NonLinearElasticity::Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l){
	// Function that returns the dot product of v1 and v2,
	// which must have the same length l
	PetscScalar result = 0.0;
	for (PetscInt i=0; i<l; i++){
		result = result + v1[i]*v2[i];
	}
	return result;
}

void NonLinearElasticity::DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar *dNdxi, PetscScalar *dNdeta){
	//differentiatedShapeFunctions - Computes differentiated shape functions
	// At the point given by (xi, eta).
	// With respect to xi:
	dNdxi[0]  = -0.25*(1.0-eta);
	dNdxi[1]  =  0.25*(1.0-eta);
	dNdxi[2]  =  0.25*(1.0+eta);
	dNdxi[3]  = -0.25*(1.0+eta);
	// With respect to eta:
	dNdeta[0] = -0.25*(1.0-xi);
	dNdeta[1] = -0.25*(1.0+xi);
	dNdeta[2] =  0.25*(1.0+xi);
	dNdeta[3] =  0.25*(1.0-xi);
	
}

PetscScalar NonLinearElasticity::Inverse2M(PetscScalar J[][2], PetscScalar invJ[][2]){
	//inverse3M - Computes the inverse of a 3x3 matrix
	PetscScalar detJ = J[0][0]*J[1][1]-J[1][0]*J[0][1];
	invJ[0][0] =  J[1][1]/detJ;
	invJ[0][1] = -J[0][1]/detJ;
	invJ[1][0] = -J[1][0]/detJ;
	invJ[1][1] =  J[0][0]/detJ;

	return detJ;
}

