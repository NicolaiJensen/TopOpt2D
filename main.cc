#include <petsc.h>
#include <TopOpt.h>
#include <LinearElasticity.h>
#include <MMA.h>
#include <Filter.h>
#include <mpi.h>
//#include <MPIIO.h>
#include <FiniteDiff.h>
#include <WriteVTS.h>
/*
 * Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 * 
 * Disclaimer:                                                              
 * The authors reserves all rights but does not guaranty that the code is   
 * free from errors. Furthermore, we shall not be liable in any event     
 * caused by the use of the program.                                     
 */

static char help[] = "2D TopOpt using KSP-MG on PETSc's DMDA (structured grids) \n";

int main(int argc, char *argv[]){
    
    // Error code for debugging
    PetscErrorCode ierr;
    
    // Initialize PETSc / MPI and pass input arguments to PETSc
    PetscInitialize(&argc,&argv,PETSC_NULL,help);
    
    // STEP 1: The Optimization Parameters, Data and Mesh
    TopOpt *opt = new TopOpt();
    
    // STEP 2: The Physics
    LinearElasticity *physics = new LinearElasticity(opt);
    
    // STEP 3: The Filtering
    Filter *filter = new Filter(opt);
    
    // STEP 4: Visualization
    WriteVTS *output = new WriteVTS(opt);
    
    // STEP 5: The optimizer MMA
    MMA *mma;
    PetscInt itr = 0;
    opt->AllocateMMAwithRestart(&itr, &mma); // allow for restart !     
    
//     //  STEP 6: FINITE DIFFERENT CHECK  
//     FiniteDiff *SenDF= new FiniteDiff(opt); 
//     ierr = SenDF->FiniteDiffCheck(opt); CHKERRQ(ierr);
//     delete SenDF;
//     return 0;
    
    // Initialize the periodization matrix
    if ( opt->MapType != 0 ) {
        opt->PeriodizationMatrix(opt); 
    }
    
    // STEP 7: Filter the initial design/restarted design
    ierr = filter->FilterProject(opt); CHKERRQ(ierr);
    
    if ( opt->MapType != 0 ) {
        Vec DTx;
        ierr = VecDuplicate(opt->xPhys, &DTx);CHKERRQ(ierr);
        ierr = VecCopy(opt->xPhys, DTx);CHKERRQ(ierr);
        ierr = MatMult(opt->DT, DTx, opt->xPhys);CHKERRQ(ierr);
        VecDestroy(&DTx);
    }  
    // Write to visualization files
    output->WritetoVTS(opt,itr);
    
    // STEP 8: OPTIMIZATION LOOP   
    PetscScalar ch = 1.0;
    double t1,t2;
    while (itr < opt->maxItr && ch > 0.00001){
        // Update iteration counter
        itr++;
        
        // start timer
        t1 = MPI_Wtime();
        
        // Compute (a) obj+const, (b) sens, (c) obj+const+sens 
        ierr = physics->ComputeObjectiveConstraintsSensitivities(opt); CHKERRQ(ierr);
        
        // Compute objective scale
        if (itr==1){ 
            opt->fscale = 100.0/opt->fx; 
        }
        // Scale objectie and sens
        opt->fx = opt->fx*opt->fscale;
        VecScale(opt->dfdx,opt->fscale);
        
        if ( opt->MapType != 0 ) {
        // Periodize design gradients
        Vec DTdfdx;
        ierr = VecDuplicate(opt->dfdx, &DTdfdx);CHKERRQ(ierr);
        ierr = VecCopy(opt->dfdx, DTdfdx);CHKERRQ(ierr);
        ierr = MatMultTranspose(opt->DT, DTdfdx, opt->dfdx);CHKERRQ(ierr);    
        ierr = VecCopy(opt->dgdx[0], DTdfdx);CHKERRQ(ierr);
        ierr = MatMultTranspose(opt->DT, DTdfdx, opt->dgdx[0]);CHKERRQ(ierr);       
        VecDestroy(&DTdfdx);
        }
        // Filter sensitivities (chainrule)
        ierr = filter->Gradients(opt); CHKERRQ(ierr);
        
        // Sets outer movelimits on design variables
        ierr = mma->SetOuterMovelimit(opt->Xmin,opt->Xmax,opt->movlim,opt->x,opt->xmin,opt->xmax); CHKERRQ(ierr);
        
        // Update design by MMA
        ierr = mma->Update(opt->x,opt->dfdx,opt->gx,opt->dgdx,opt->xmin,opt->xmax); CHKERRQ(ierr);
        
        // Inf norm on the design change
        ch = mma->DesignChange(opt->x,opt->xold);
        
        // Filter design field
        ierr = filter->FilterProject(opt); CHKERRQ(ierr);
        
        if ( opt->MapType != 0 ) {
        // Periodize design filtered design variables
        Vec DTx;
        ierr = VecDuplicate(opt->xPhys, &DTx);CHKERRQ(ierr);
        ierr = VecCopy(opt->xPhys, DTx);CHKERRQ(ierr);
        ierr = MatMult(opt->DT, DTx, opt->xPhys);CHKERRQ(ierr);
        VecDestroy(&DTx);
        }
        /*
         *         	// Global coordinates and a pointer                                            ???
         *	Vec lcoor; // borrowed ref - do not destroy!
         *	PetscScalar *lcoorp;
         *        
         *	// Get local coordinates in local node numbering including ghosts       
         *	ierr = DMGetCoordinatesLocal(opt->da_elem,&lcoor); CHKERRQ(ierr);
         *	VecGetArray(lcoor,&lcoorp); // Putting local coordinates into pointer
         *        
         *	// Get local dof number
         *	PetscInt nn;
         *	VecGetSize(lcoor,&nn); 
         *	// Set the values of boundaries and forces:
         *        PetscPrintf(PETSC_COMM_WORLD,"XC3: %.2f\n",
         *                    opt->xc[3]);
         *	for ( PetscInt i=0;i<nn;i++ ){
         *            // Set fixed-fixed bottom boundary
         *            if ( (i % 2 == 0 && (lcoorp[i+1] <= opt->xc[3]/20.0)) || (i % 2 == 0 && (lcoorp[i+1] >= opt->xc[3]-opt->xc[3]/20.0))  ){
         *                VecSetValue(opt->x,i/2,1.0,INSERT_VALUES);
         *                VecSetValue(opt->xPhys,i/2,1.0,INSERT_VALUES);
    }
    } 
    VecAssemblyBegin(opt->x);
    VecAssemblyEnd(opt->x);
    VecAssemblyBegin(opt->xPhys);
    VecAssemblyEnd(opt->xPhys);*/
        // stop timer
        t2 = MPI_Wtime();
        
        // Print to screen
        PetscPrintf(PETSC_COMM_WORLD,"It.: %i, obj.: %f, g[0]: %f, ch.: %f, time: %f\n",
                    itr,opt->fx,opt->gx[0], ch,t2-t1);
        
        // Write to visualization files
//         if (itr<11 || itr%10==0){
//             output->WritetoVTS(opt,itr);
//         }
        
        // Dump data needed for restarting code at termination
        if (itr%3==0)	{
            opt->WriteRestartFiles(&itr, mma);
            physics->WriteRestartFiles();
        }
    }
    // Write restart files
    opt->WriteRestartFiles(&itr, mma);  
    physics->WriteRestartFiles();
    
    //Dump final design
    output->WritetoVTS(opt,itr);
    
    // STEP 7: CLEAN UP AFTER YOURSELF
    delete mma;
    delete output;
    delete filter;
    delete opt;  
    delete physics;
    
    // Finalize PETSc / MPI
    PetscFinalize();
    return 0;
}
