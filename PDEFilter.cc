#include <PDEFilter.h>
#include <TopOpt.h>
//#include <petsc-private/dmdaimpl.h>
#include <petsc/private/dmdaimpl.h>
/* -----------------------------------------------------------------------------
Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013 
Copyright (C) 2013-2014,

This PDEFilter implementation is licensed under Version 2.1 of the GNU
Lesser General Public License.  

This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This Module is distributed in the hope that it will be useful,implementation 
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */


PDEFilt::PDEFilt(TopOpt *opt)
{

    
      
      
	R=opt->rmin/2.0/sqrt(3); // conversion factor for the PDEfilter

	nlvls=3; // MG levels

	// number of nodal dofs
	PetscInt numnodaldof = 1;

	// Stencil width: each node connects to a box around it - linear elements
	PetscInt stencilwidth = 1;

	PetscScalar dx,dy;
	DMBoundaryType bx, by;
	DMDAStencilType stype;
	{
		// Extract information from the nodal mesh
		PetscInt M,N,md,nd; 
		DMDAGetInfo(opt->da_nodes,NULL,&M,&N,NULL,&md,&nd,NULL,NULL,NULL,&bx,&by,NULL,&stype); 

		// Find the element size
		Vec lcoor;
		DMGetCoordinatesLocal(opt->da_nodes,&lcoor);
		PetscScalar *lcoorp;
		VecGetArray(lcoor,&lcoorp);

		PetscInt nel, nen;
		const PetscInt *necon;
                DMDAGetElements_2D(opt->da_nodes,&nel,&nen,&necon);

                
                  
		// Use the first element to compute the dx, dy, dz
		dx = lcoorp[2*necon[0*nen + 1]+0]-lcoorp[2*necon[0*nen + 0]+0];
		dy = lcoorp[2*necon[0*nen + 2]+1]-lcoorp[2*necon[0*nen + 1]+1];
		 
		VecRestoreArray(lcoor,&lcoorp);

                 
		// ELement volume
	 
		elemVol = dx*dy;
                

		nn[0]=M;
		nn[1]=N;
	 

		ne[0]=nn[0]-1; 
		ne[1]=nn[1]-1; 
	 


		xc[0]=0.0;
		xc[1]=ne[0]*M;
		xc[2]=0.0;
		xc[3]=ne[1]*N;

	}

	// Create the nodal mesh
	DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nn[0],nn[1],PETSC_DECIDE,PETSC_DECIDE,
			numnodaldof,stencilwidth,0,0,&(da_nodal));
	// Set the coordinates
	DMDASetUniformCoordinates(da_nodal, xc[0],xc[1], xc[2],xc[3], 0.0,0.0);
	// Set the element type to Q1: Otherwise calls to GetElements will change to P1 !
	// STILL DOESN*T WORK !!!!
	DMDASetElementType(da_nodal, DMDA_ELEMENT_Q1);

	//Create the element mesh

	// find the geometric partitioning of the nodal mesh, so the element mesh will coincide
	PetscInt md,nd;
	DMDAGetInfo(da_nodal,NULL,NULL,NULL,NULL,&md,&nd,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
	PetscInt *Lx=new PetscInt[md];
	PetscInt *Ly=new PetscInt[nd];
 
	// get number of nodes for each partition
	const PetscInt *LxCorrect, *LyCorrect ;
	DMDAGetOwnershipRanges(da_nodal, &LxCorrect, &LyCorrect, NULL);
	// subtract one from the lower left corner
	for (int i=0; i<md; i++){
		Lx[i] = LxCorrect[i];
		if (i==0){Lx[i] = Lx[i]-1;}
	}
	for (int i=0; i<nd; i++){
		Ly[i] = LyCorrect[i];
		if (i==0){Ly[i] = Ly[i]-1;}
	}
	 
	PetscInt overlap=0; 
	// Create the element grid:
	DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nn[0]-1,nn[1]-1,md,nd,
			1,overlap,Lx,Ly,&(da_element));


	delete [] Lx;
	delete [] Ly;
        
         
                 
       PDEFilterMatrix2D(dx,dy,R, KF, TF);

	 

	//create the stiffness matrix
	DMCreateMatrix(da_nodal,&(K));
	//create RHS
	DMCreateGlobalVector(da_nodal,&(RHS));
	DMCreateGlobalVector(da_element,&(X));
	VecDuplicate(RHS, &U);


	//Create T matrix
	{
		PetscInt m;
		PetscInt n;
		//PetscInt M;
		//PetscInt N;

		//m,M  extract it from RHS
		//n,N  extract it from X
		VecGetLocalSize(RHS,&m);
		VecGetLocalSize(X,&n);

		MatCreateAIJ(PETSC_COMM_WORLD , m, n, PETSC_DETERMINE, PETSC_DETERMINE, 4, NULL,3,NULL, &T);

		ISLocalToGlobalMapping rmapping;
		ISLocalToGlobalMapping cmapping;	

		DMGetLocalToGlobalMapping(da_nodal, &rmapping);	
		DMGetLocalToGlobalMapping(da_element, &cmapping);


		MatSetLocalToGlobalMapping(T,rmapping,cmapping);

	}


	MatAssemble();
	SetUpSolver();

	//test
	PetscRandom rctx;
	PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
	PetscRandomSetType(rctx,PETSCRAND48);
	VecSetRandom(X,rctx);
	PetscRandomDestroy(&rctx);

	FilterProject(X,X);
	Gradients(X,X);

	//
	PetscPrintf(PETSC_COMM_WORLD,"Done setting up the PDEFilter\n");
}

PetscErrorCode PDEFilt::FilterProject(Vec OX, Vec FX)
{

	PetscErrorCode ierr;

	double t1,t2;
	PetscScalar rnorm;
	PetscInt niter;

	t1 = MPI_Wtime();
	ierr = MatMult(T,OX,RHS); CHKERRQ(ierr);
	ierr = VecCopy(RHS,U); CHKERRQ(ierr);
	ierr = VecScale(RHS,elemVol);CHKERRQ(ierr);
	ierr = KSPSolve(ksp,RHS,U); CHKERRQ(ierr);
	ierr = KSPGetIterationNumber(ksp,&niter); CHKERRQ(ierr);
	ierr = KSPGetResidualNorm(ksp,&rnorm);  CHKERRQ(ierr);
	ierr = MatMultTranspose(T,U,FX); CHKERRQ(ierr);

	t2 = MPI_Wtime();
	PetscPrintf(PETSC_COMM_WORLD,"PDEFilter solver:  iter: %i, rerr.: %e, time: %f\n",niter,rnorm,t2-t1);
	return ierr;
}

PetscErrorCode PDEFilt::Gradients(Vec OS, Vec FS)
{
	return FilterProject(OS,FS);
}

PDEFilt::~PDEFilt()
{
	Free();
}

PetscErrorCode PDEFilt::Free()
{

	PetscErrorCode ierr;

	KSPDestroy(&ksp);

	VecDestroy(&RHS);
	VecDestroy(&X);
	VecDestroy(&U);

	MatDestroy(&T);
	MatDestroy(&K);

	ierr=DMDestroy(&da_nodal);    CHKERRQ(ierr);
	ierr=DMDestroy(&da_element);  CHKERRQ(ierr);

	return ierr;

}

void PDEFilt::MatAssemble()
{
	// Get the FE mesh structure (from the nodal mesh)
	PetscInt nel, nen;
	const PetscInt *necon;
	DMDAGetElements_2D(da_nodal,&nel,&nen,&necon);
	MatZeroEntries(K);
	MatZeroEntries(T);
        
        PetscInt *edof = new PetscInt[4];
		
		
		for (PetscInt i=0;i<nel;i++)
		{
			// loop over element nodes
			//PetscPrintf(PETSC_COMM_WORLD, "El %i, nodes: ", i);
			for (PetscInt j=0;j<nen;j++)
			{
				edof[j]=necon[i*nen+j];
				//PetscPrintf(PETSC_COMM_WORLD, "%i, ", gidx[edof[j]]);
			}
			//PetscPrintf(PETSC_COMM_WORLD, "\n");
			
		      MatSetValuesLocal(K,4,edof,4,edof,KF,ADD_VALUES); 
			// Assemble the T matrix 
                      MatSetValuesLocal(T,4,edof,1,&i,TF,ADD_VALUES); 
			
                        
 


                }
	MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(T, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(T, MAT_FINAL_ASSEMBLY);

	delete [] edof;

}


PetscErrorCode PDEFilt::SetUpSolver()
{
	//make sure ksp is not allocated before 
	PetscErrorCode ierr;
	PC pc;

	// The fine grid Krylov method
	KSPCreate(PETSC_COMM_WORLD,&ksp);
	ierr = KSPSetType(ksp,KSPFGMRES); // KSPCG, KSPGMRES
	PetscInt restart = 20;
	ierr = KSPGMRESSetRestart(ksp,restart);

	PetscScalar rtol = 1.0e-8;
	PetscScalar atol = 1.0e-50;
	PetscScalar dtol = 1.0e3;
	PetscInt maxitsGlobal = 60;
	ierr = KSPSetTolerances(ksp,rtol,atol,dtol,maxitsGlobal);
	ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
	KSPSetOperators(ksp,K,K); // ,SAME_PRECONDITIONER is now set in the prec

	//preconditioner
	KSPGetPC(ksp,&pc);
	PCSetType(pc,PCMG);
	// Set solver from options
	KSPSetFromOptions(ksp);
	// Get the prec again - check if it has changed
	KSPGetPC(ksp,&pc);
	ierr = PCSetReusePreconditioner(pc,PETSC_TRUE); CHKERRQ(ierr);
	// Flag for pcmg pc
	PetscBool pcmg_flag = PETSC_TRUE;
	PetscObjectTypeCompare((PetscObject)pc,PCMG,&pcmg_flag);
	// Only if PCMG is used
	if (pcmg_flag){
		// DMs for grid hierachy
		DM  *da_list,*daclist;
		Mat R;
		PetscMalloc(sizeof(DM)*nlvls,&da_list);
		for (PetscInt k=0; k<nlvls; k++) da_list[k] = NULL;
		PetscMalloc(sizeof(DM)*nlvls,&daclist);
		for (PetscInt k=0; k<nlvls; k++) daclist[k] = NULL;
		// Set 0 to the finest level
		daclist[0] = da_nodal;

		// Coordinates
		PetscReal xmin=xc[0], xmax=xc[1], ymin=xc[2], ymax=xc[3];

		// Set up the coarse meshes
		ierr=DMCoarsenHierarchy(da_nodal, nlvls-1,&daclist[1]); CHKERRQ(ierr);
		for (PetscInt k=0; k<nlvls; k++) {
			// NOTE: finest grid is nlevels - 1: PCMG MUST USE THIS ORDER ???
			da_list[k] = daclist[nlvls-1-k];
			DMDASetUniformCoordinates(da_list[k],xmin,xmax,ymin,ymax,0.0,0.0);
		}

		// the PCMG specific options
		PCMGSetLevels(pc,nlvls,NULL);
		PCMGSetType(pc,PC_MG_MULTIPLICATIVE); // Default
		PCMGSetCycleType(pc,PC_MG_CYCLE_V);
		PCMGSetGalerkin(pc,PETSC_TRUE);
		for (PetscInt k=1; k<nlvls; k++) {
			DMCreateInterpolation(da_list[k-1],da_list[k],&R,NULL);
			PCMGSetInterpolation(pc,k,R);
			MatDestroy(&R);
		}

		for (PetscInt k=1; k<nlvls; k++) { //0 level should be dealocated in the destructor
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
			// PetscInt restarts[nlvls] = {10, 1 , 1}; // coarse .... fine
			restart = 10;
			ierr = KSPGMRESSetRestart(cksp,restart);
			rtol = 1.0e-8;
			atol = 1.0e-50;
			dtol = 1e3;
			PetscInt maxits = 10;
			ierr = KSPSetTolerances(cksp,rtol,atol,dtol,maxits);
			// The preconditioner
			PC cpc;
			KSPGetPC(cksp,&cpc);
			//PCSetType(cpc,PCSOR); // PCSOR, PCSPAI (NEEDS TO BE COMPILED), PCJACOBI
			PCSetType(cpc,PCJACOBI);

			// Set smoothers on all levels (except for coarse grid):
			for (PetscInt k=1;k<nlvls;k++){
				KSP dksp;
				PCMGGetSmoother(pc,k,&dksp);
				PC dpc;
				KSPGetPC(dksp,&dpc);
				ierr = KSPSetType(dksp,KSPGMRES); // KSPCG, KSPGMRES, KSPCHEBYSHEV (VERY GOOD FOR SPD)
				restart = 1;
				ierr = KSPGMRESSetRestart(dksp,restart);
				ierr = KSPSetTolerances(dksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,restart); // NOTE maxitr=restart;
				PCSetType(dpc,PCJACOBI);// PCJACOBI, PCSOR for KSPCHEBYSHEV very good
			}
		}


	}


	// 	// Write check to screen:
	//         // Check the overall Krylov solver
	//         KSPType ksptype;
	//         KSPGetType(ksp,&ksptype);
	//         PCType pctype;
	//         PCGetType(pc,&pctype);
	//         PetscInt mmax;
	//         KSPGetTolerances(ksp,NULL,NULL,NULL,&mmax);
	//         PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
	//         PetscPrintf(PETSC_COMM_WORLD,"################# Linear solver settings #####################\n");
	//         PetscPrintf(PETSC_COMM_WORLD,"# Main solver: %s, prec.: %s, maxiter.: %i \n",ksptype,pctype,mmax);
	// 
	//         // Only if pcmg is used
	//         if (pcmg_flag){
	//                 // Check the smoothers and coarse grid solver:
	//                 for (PetscInt k=0;k<nlvls;k++){
	//                         KSP dksp;
	//                         PC dpc;
	//                         KSPType dksptype;
	//                         PCMGGetSmoother(pc,k,&dksp);
	//                         KSPGetType(dksp,&dksptype);
	//                         KSPGetPC(dksp,&dpc);
	//                         PCType dpctype;
	//                         PCGetType(dpc,&dpctype);
	//                         PetscInt mmax;
	//                         KSPGetTolerances(dksp,NULL,NULL,NULL,&mmax);
	//                         PetscPrintf(PETSC_COMM_WORLD,"# Level %i smoother: %s, prec.: %s, sweep: %i \n",k,dksptype,dpctype,mmax);
	//                 }
	//         }
	//         PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");

 	return(0);
	 
}



PetscErrorCode PDEFilt::DMDAGetElements_2D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]) {
	PetscErrorCode ierr;
	DM_DA          *da = (DM_DA*)dm->data;
	PetscInt       i,xs,xe,Xs,Xe;
	PetscInt       j,ys,ye,Ys,Ye;
	PetscInt       cnt=0, cell[4], ns=1, nn=4;
	PetscInt       c;
	if (!da->e) {
		if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=4;}
		ierr = DMDAGetCorners(dm,&xs,&ys,NULL,&xe,&ye,NULL);CHKERRQ(ierr);
		ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,NULL,&Xe,&Ye,NULL);CHKERRQ(ierr);
		xe    += xs; Xe += Xs; if (xs != Xs) xs -= 1;
		ye    += ys; Ye += Ys; if (ys != Ys) ys -= 1;
		da->ne = ns*(xe - xs - 1)*(ye - ys - 1);
		PetscMalloc((1 + nn*da->ne)*sizeof(PetscInt),&da->e);
		
		for (j=ys; j<ye-1; j++) {
			for (i=xs; i<xe-1; i++) {
				cell[0] = (i-Xs  ) + (j-Ys  )*(Xe-Xs) ;
				cell[1] = (i-Xs+1) + (j-Ys  )*(Xe-Xs) ;
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

void PDEFilt::PDEFilterMatrix2D(PetscScalar dx, PetscScalar dy,
				PetscScalar RR,PetscScalar *KK, PetscScalar *T)
{
	
	PetscScalar HLeta=dx/2.0;
	PetscScalar HLxi=dy/2.0;
	
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626}; 
	// Corresponding weights
	PetscScalar W[2] = {1.0, 1.0};
	
	PetscScalar C[2][2] = {{RR*RR, 0.0},
	{ 0.0, RR*RR}};
	
	//
	
	memset(KK, 0, sizeof(KK[0])*4*4); // zero out
	memset(T, 0, sizeof(T[0])*4); // zero out
	
	PetscScalar dNdxi[4]; PetscScalar dNdeta[4]; PetscScalar NN[4];
	PetscScalar B[2][4]; // Note: Small enough to be allocated on stack
	
	// Perform the numerical integration
	for (PetscInt ii=0; ii<2; ii++){
		for (PetscInt jj=0; jj<2; jj++){
			
			// Integration point
			PetscScalar xi = GP[ii]*HLxi; 
			PetscScalar eta = GP[jj]*HLeta; 
			
			DifferentiatedShapeFunctions(xi, eta, dx,dy,dNdxi, dNdeta);
			ShapeFunctions(xi, eta,dx,dy,NN);
			memset(B, 0, sizeof(B[0][0])*2*4); // zero out
			PetscScalar weight = W[ii]*W[jj]*HLxi*HLeta;
			
			for (PetscInt i=0; i<4; i++){
				B[0][i]=dNdxi[i];
				B[1][i]=dNdeta[i];
				
				
			}
			// Finally, add to the element matrix
			for (PetscInt i=0; i<4; i++){
				for (PetscInt j=0; j<4; j++){
					for (PetscInt k=0; k<2; k++){
						for (PetscInt l=0; l<2; l++){	
							KK[j+4*i] = KK[j+4*i] + weight*(B[k][i] * C[k][l] * B[l][j]);	
						}
					}
					KK[j+4*i]= KK[j+4*i] + weight*NN[i]*NN[j];
				}
			}
			
		}
	}
	
	PetscScalar vol=1.0;
	T[0]=0.25*vol;
	T[1]=0.25*vol;
	T[2]=0.25*vol;
	T[3]=0.25*vol;
	
	
	
}
void PDEFilt::DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar FLxi, PetscScalar FLeta, PetscScalar *dNdxi, PetscScalar *dNdeta)
{
	//differentiatedShapeFunctions - Computes differentiated shape functions
	// At the point given by (xi, eta, zeta).
	// With respect to xi:
	PetscScalar HLeta=FLeta/2.0;
	PetscScalar HLxi=FLxi/2.0;
	
	dNdxi[0]  = -0.25*(1.0-eta/HLeta)/HLxi;
	dNdxi[1]  =  0.25*(1.0-eta/HLeta)/HLxi;
	dNdxi[2]  =  0.25*(1.0+eta/HLeta)/HLxi;
	dNdxi[3]  = -0.25*(1.0+eta/HLeta)/HLxi;
	
	// With respect to eta:
	dNdeta[0] = -0.25*(1.0-xi/HLxi)/HLeta;
	dNdeta[1] = -0.25*(1.0+xi/HLxi)/HLeta;
	dNdeta[2] =  0.25*(1.0+xi/HLxi)/HLeta;
	dNdeta[3] =  0.25*(1.0-xi/HLxi)/HLeta;
	
	
}



void PDEFilt::ShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar FLxi, PetscScalar FLeta, PetscScalar *NN){
	//differentiatedShapeFunctions - Computes differentiated shape functions
	// At the point given by (xi, eta, zeta).
	// With respect to xi:
	PetscScalar HLeta=FLeta/2.0;
	PetscScalar HLxi=FLxi/2.0;
	
	NN[0]  = 0.25*(1.0-xi/HLxi)*(1.0-eta/HLeta);
	NN[1]  =  0.25*(1.0+xi/HLxi)*(1.0-eta/HLeta);
	NN[2]  =  0.25*(1.0+xi/HLxi)*(1.0+eta/HLeta);
	NN[3]  =  0.25*(1.0-xi/HLxi)*(1.0+eta/HLeta);
	
	
}

