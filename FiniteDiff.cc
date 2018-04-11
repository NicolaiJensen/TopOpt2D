#include <FiniteDiff.h>

FiniteDiff::FiniteDiff(TopOpt *opt){
    
    
}

FiniteDiff::~FiniteDiff(){
    
}

PetscErrorCode FiniteDiff::FiniteDiffCheck(TopOpt *opt){
    PetscErrorCode ierr;
    // STEP 2: THE PHYSICS
    
    LinearElasticity *physics = new LinearElasticity(opt);
    
    
    // STEP 3: THE FILTERING
    Filter *filter = new Filter(opt);	
    opt->PeriodizationMatrix(opt); 
    
    // Filter density variable to get xPhys	
    
    
    ierr = filter->FilterProject(opt); CHKERRQ(ierr);
    
//     Vec DTx;
//     ierr = VecDuplicate(opt->xPhys, &DTx);CHKERRQ(ierr);
//     ierr = VecCopy(opt->xPhys, DTx);CHKERRQ(ierr);
//     ierr = MatMult(opt->DT, DTx, opt->xPhys);CHKERRQ(ierr);
//     VecDestroy(&DTx);
    
    
    // Compute (a) obj+const, (b) sens, (c) obj+const+sens 
    // 	ierr = physics->ComputeObjectiveConstraints(opt); CHKERRQ(ierr);
    // 	ierr = physics->ComputeSensitivities(opt); CHKERRQ(ierr);
    
    ierr = physics->ComputeObjectiveConstraintsSensitivities(opt); CHKERRQ(ierr);
    
    for (PetscInt jj=0;jj<opt->m;jj++){
        PetscPrintf(PETSC_COMM_WORLD,"Con:  %i, gx: %f  ",jj,opt->gx[jj]);
    }
    
    PetscPrintf(PETSC_COMM_WORLD,"\n\n ");
    
    
    // 	 ierr = physics->ComputeObjectiveConstraints(opt); CHKERRQ(ierr);
    // 	 
    // 	 for (PetscInt jj=0;jj<opt->m;jj++){
    // 		 PetscPrintf(PETSC_COMM_WORLD,"Con:  %i, gx:  %f  ",jj,opt->gx[jj]);
    // 	 }
    // 	 
    PetscPrintf(PETSC_COMM_WORLD,"\n\n ");
    
//     // PERIODIZE
//     Vec DTdfdx;
//     ierr = VecDuplicate(opt->dfdx, &DTdfdx);CHKERRQ(ierr);
//     ierr = VecCopy(opt->dfdx, DTdfdx);CHKERRQ(ierr);
//     ierr = MatMultTranspose(opt->DT, DTdfdx, opt->dfdx);CHKERRQ(ierr);    
//     
//     
//     ierr = VecCopy(opt->dgdx[0], DTdfdx);CHKERRQ(ierr);
//     ierr = MatMultTranspose(opt->DT, DTdfdx, opt->dgdx[0]);CHKERRQ(ierr);       
//     VecDestroy(&DTdfdx);
//     
    // filtering gradients
    ierr = filter->Gradients(opt); CHKERRQ(ierr);
    
    PetscPrintf(PETSC_COMM_WORLD,"Finite difference check!\n");	
    
    
    //
    
    PetscInt NumVar=7;
    PetscInt  NumID=4;
    
    PetscScalar ddx[7]={1.0e-1,1.0e-2,1.0e-3,1.0e-4,1.0e-5,1.0e-6,1e-7};
    PetscInt  IDloc[4]={1,1,1,1}; // local ID's 
    PetscInt  IDproc[4]={0,1,2,3}; // procesoor ID -> linked VERY MUCH to IDloc
    
    // Get pointer to objective sens
    PetscScalar *df;
    VecGetArray(opt->dfdx,&df);
    
    
    // Backp objective
    PetscScalar f0=opt->fx;
    PetscScalar *g0=new PetscScalar[opt->m];
    
    for(PetscInt i=0; i<opt->m;i++){
        g0[i]=opt->gx[i];
    }
    
    // Backup design variable (unfiltered)
    Vec x0;
    VecDuplicate(opt->x,&(x0));
    VecCopy(opt->x,x0);
    
    // Get rank
    PetscMPIInt  rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    for (PetscInt i=0; i<NumID; i++){
        
        // If rank check here - only one core will compute !!!!
        
        for (PetscInt j=0; j<NumVar; j++){
            // Reset design vector
            VecCopy(x0,opt->x);
            
            // Get pointer to design vcetor
            PetscScalar *xp;
            VecGetArray(opt->x,&xp);
            
            // perturb on a single proc
            if (rank==IDproc[i]){
                xp[IDloc[i]]+=ddx[j];
            }
            VecRestoreArray(opt->x,&xp);
            // filter the design
            ierr = filter->FilterProject(opt); CHKERRQ(ierr);
            
//             Vec DTx;
//             ierr = VecDuplicate(opt->xPhys, &DTx);CHKERRQ(ierr);
//             ierr = VecCopy(opt->xPhys, DTx);CHKERRQ(ierr);
//             ierr = MatMult(opt->DT, DTx, opt->xPhys);CHKERRQ(ierr);
//             VecDestroy(&DTx);
            
            
            // Compute objective
            ierr = physics->ComputeObjectiveConstraints(opt); CHKERRQ(ierr);
            
            //ierr = ConlenRob->EnforLen(opt,opt->xInt,opt->Ceta,opt->beta,0); CHKERRQ(ierr);
            
            
            // Print out
            if (rank==IDproc[i]){
                
                
                std::cout<<"IDProc:  "<<rank<<std::endl;
                    PetscScalar *df;
                 VecGetArray(opt->dfdx,&df);
                    PetscScalar dfFD = (opt->fx-f0)/ddx[j];
                    PetscScalar dfREF =  df[IDloc[i]];
                    PetscScalar dfERR= (dfFD-dfREF)/dfREF*100.0;
                    std::cout<<rank<<"  Obj:  "<<f0<<" Aft. Perb. "<<opt->fx<<", dfFD: "<<dfFD<<", dfREF; "<<dfREF<<", RelErr: "<<dfERR<<std::endl;
                    VecRestoreArray(opt->dfdx,&df);
                
                for (PetscInt jj=0;jj<opt->m;jj++){
                    PetscScalar dgFD = (opt->gx[jj]-g0[jj])/ddx[j];
                    PetscScalar *dg;
                
                    VecGetArray(opt->dgdx[jj],&dg);
                    
                    PetscScalar dgREF =  dg[IDloc[i]];
                    PetscScalar dgERR= (dgFD-dgREF)/dgREF*100.0;
                    std::cout<<rank<<"  OldCon: "<<jj<<":   "<<g0[jj]<<" Newcon "<<opt->gx[jj]<<", dgFD: "<<dgFD<<", dgREF; "<<dgREF<<", RelErr: "<<dgERR<<std::endl;
                    VecRestoreArray(opt->dgdx[jj],&dg);
                    
                   
                }
                
            }
            //PetscPrintf(PETSC_COMM_WORLD,"Obj: %f  Con:  %f\n",tmpf,tmpg);
            
            
        }
    }
    
    
    VecDestroy(&x0);	
    
    PetscPrintf(PETSC_COMM_WORLD,"Finite difference check finished!\n");	
    delete [] g0;
    delete filter;
    delete physics;
    
    return ierr;
    
    
}



