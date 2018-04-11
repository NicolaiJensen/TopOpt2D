#include <TopOpt.h>
#include <cmath>  
#include <cstdlib>
#include <ctime>
/*
 * Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013  Convert to 2D by Fengwen Wang
 * 
 * Disclaimer:                                                              
 * The authors reserves all rights but does not guaranty that the code is   
 * free from errors. Furthermore, we shall not be liable in any event     
 * caused by the use of the program.                                     
 */

TopOpt::TopOpt(){
    
    x = NULL;
    xPhys = NULL;
    dfdx = NULL;
    dgdx = NULL;
    gx = NULL;
    da_nodes = NULL;
    da_elem = NULL;
    xo1 = NULL;
    xo2 = NULL;
    U   = NULL;
    L   = NULL;
    
    SetUp();
}

TopOpt::~TopOpt(){
    
    // Delete vectors
    if (x != NULL){ VecDestroy(&x); }
    if (xPhys != NULL){ VecDestroy(&xPhys); }
    if (dfdx != NULL){ VecDestroy(&dfdx); }
    if (dgdx != NULL){ VecDestroyVecs(m,&dgdx); }
    if (xold != NULL){ VecDestroy(&xold); }
    if (xmin != NULL){ VecDestroy(&xmin); }
    if (xmax != NULL){ VecDestroy(&xmax); }
    if (da_nodes != NULL){ DMDestroy(&(da_nodes)); }
    if (da_elem != NULL){ DMDestroy(&(da_elem)); }
    
    // Delete constraints
    if (gx != NULL){ delete [] gx; }
    
    // Mma restart method    		
    if (xo1 != NULL){ VecDestroy(&xo1); }
    if (xo2 != NULL){ VecDestroy(&xo2); }
    if (L != NULL){ VecDestroy(&L); }
    if (U != NULL){ VecDestroy(&U); }
}

PetscErrorCode TopOpt::SetUp(){
    PetscErrorCode ierr;
    
    // SET DEFAULTS for FE mesh and levels for MG solver
    nxyz[0] = 129; // Number of nodes in x
    nxyz[1] = 129; // Number of nodes in y
    xc[0] = 0.0;   // x-min boundary
    xc[1] = 1.0;   // x-max boundary
    xc[2] = 0.0;   // y-min boundary
    xc[3] = 1.0;   // y-max boundary
    nlvls = 4;     // Number of levels for MG
    
    // SET DEFAULTS for optimization problems
    volfrac = 0.5; // Volume fraction
    maxItr = 400; // Maximum iterations
    rmin = 0.01;   // Filter radius
    penal = 7.0;   // Penalization factor
    Emin = 1.0e-9   ; // Minimum elasticity modulus for "void" regions
    Emax = 1.0;    // Maximum elasticity modulus
    F0 = 1.0;      // Force 
    nu = 0.3;      // Poisson's ratio
    AssType = 1;   // Type of stress-strain:
                        // 0: Plane strain
                        // 1: Plain stress 
    filter = 1;    // Filter type:
                        // 0: Sensitivity filtering
                        // 1: Densitity filtering
                        // 2: PDE filtering
                        // others: NO FILTERING
    m = 1;         // Volume constraint
    // SET DEFAULT MMA CONSTRAINTS
    Xmin = 0.0;    // MMA density minimum
    Xmax = 1.0;    // MMA density maximum
    movlim = 0.2;  // Move limits for MMA
    
    // SET DEFAULT MAPPING OPTIONS
    divx = 1;       // Number of divisions in x
    divy = 1;       // Number of divisions in y
    MapType = 4;    // Map type:
                        // 0: No mapping
                        // 1: Unit cell mapping
                        // 2: Unit cell mapping w. x-symmetry
                        // 3: Unit cell mapping w. y-symmetry
                        // 4: Unit cell mapping w. x- and y-symmetry
                        // 5: Unit cell mapping with diagonal symmetry (8 sub-domains)
    
    restart = PETSC_TRUE;
    
    ierr = SetUpMESH(); CHKERRQ(ierr);
    ierr = SetUpOPT(); CHKERRQ(ierr);
    
    return(ierr);
}

PetscErrorCode TopOpt::SetUpMESH(){
    
    PetscErrorCode ierr;
    
    // Read input from arguments
    PetscBool flg;
    
    // Physics parameters
    PetscOptionsGetInt(NULL,NULL,"-nx",&(nxyz[0]),&flg);
    PetscOptionsGetInt(NULL,NULL,"-ny",&(nxyz[1]),&flg);
    PetscOptionsGetReal(NULL,NULL,"-xcmin",&(xc[0]),&flg);	
    PetscOptionsGetReal(NULL,NULL,"-xcmax",&(xc[1]),&flg);
    PetscOptionsGetReal(NULL,NULL,"-ycmin",&(xc[2]),&flg);
    PetscOptionsGetReal(NULL,NULL,"-ycmax",&(xc[3]),&flg);
    PetscOptionsGetInt(NULL,NULL,"-divx",&divx,&flg);
    PetscOptionsGetInt(NULL,NULL,"-divy",&divy,&flg);
    PetscOptionsGetReal(NULL,NULL,"-penal",&penal,&flg);
    PetscOptionsGetInt(NULL,NULL,"-nlvls",&nlvls,&flg);
    
    // Write parameters for the physics _ OWNED BY TOPOPT
    PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");
    PetscPrintf(PETSC_COMM_WORLD,"############################ FEM settings ##############################\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Number of nodes: (-nx,-ny,-nz):        (%i,%i) \n",nxyz[0],nxyz[1]);
    PetscPrintf(PETSC_COMM_WORLD,"# Number of degree of freedom:           %i \n",2*nxyz[0]*nxyz[1]);
    PetscPrintf(PETSC_COMM_WORLD,"# Number of elements:                    (%i,%i) \n",nxyz[0]-1,nxyz[1]-1);
    PetscPrintf(PETSC_COMM_WORLD,"# Dimensions: (-xcmin,-xcmax,..): (%f,%f)\n",xc[1]-xc[0],xc[3]-xc[2]);
    PetscPrintf(PETSC_COMM_WORLD,"# -nlvls: %i\n",nlvls);
    PetscPrintf(PETSC_COMM_WORLD,"# Number of divisions in x: %i and y: %i \n",divx,divy);
    PetscPrintf(PETSC_COMM_WORLD,"########################################################################\n");
    
    // Check if the mesh supports the chosen number of MG levels
    PetscScalar divisor = PetscPowScalar(2.0,(PetscScalar)nlvls-1.0);
    // x - dir
    if ( std::floor((PetscScalar)(nxyz[0]-1)/divisor) != (nxyz[0]-1.0)/((PetscInt)divisor) ) {
        PetscPrintf(PETSC_COMM_WORLD,"MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD,"X - number of nodes %i is cannot be halfened %i times\n",nxyz[0],nlvls-1);
        exit(0);
    }	
    // y - dir
    if ( std::floor((PetscScalar)(nxyz[1]-1)/divisor) != (nxyz[1]-1.0)/((PetscInt)divisor) ) {
        PetscPrintf(PETSC_COMM_WORLD,"MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD,"Y - number of nodes %i is cannot be halfened %i times\n",nxyz[1],nlvls-1);
        exit(0);
    }
    
    // Start setting up the FE problem
    // Boundary types: DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED, DMDA_BOUNDARY_PERIODIC
    DMBoundaryType bx = DM_BOUNDARY_NONE;
    DMBoundaryType by = DM_BOUNDARY_NONE;
    
    // Stencil type - box since this is closest to FEM (i.e. STAR is FV/FD)
    DMDAStencilType  stype = DMDA_STENCIL_BOX;
    
    // Discretization: nodes:
    // For standard FE - number must be odd
    // FOr periodic: Number must be even
    PetscInt nx = nxyz[0];
    PetscInt ny = nxyz[1];
    
    // number of nodal dofs
    PetscInt numnodaldof = 2;
    
    // Stencil width: each node connects to a box around it - linear elements
    PetscInt stencilwidth = 1;
    
    // Coordinates and element sizes: note that dx,dy,dz are half the element size
    PetscReal xmin=xc[0], xmax=xc[1], ymin=xc[2], ymax=xc[3] ;
    dx = (xc[1]-xc[0])/(PetscScalar(nxyz[0]-1));
    dy = (xc[3]-xc[2])/(PetscScalar(nxyz[1]-1));
    
    
    // Create the nodal mesh
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nx,ny,PETSC_DECIDE,PETSC_DECIDE,
                        numnodaldof,stencilwidth,0,0,&(da_nodes));
    CHKERRQ(ierr);
    
    // 	// Initialize
    //         DMSetFromOptions(da_nodes);
    //         DMSetUp(da_nodes);
    
    // Set the coordinates
    ierr = DMDASetUniformCoordinates(da_nodes, xmin,xmax, ymin,ymax, 0.0,0.0);
    CHKERRQ(ierr);
    
    // Set the element type to Q1: Otherwise calls to GetElements will change to P1 !
    // STILL DOESN*T WORK !!!!
    ierr = DMDASetElementType(da_nodes, DMDA_ELEMENT_Q1);
    CHKERRQ(ierr);
    
    // Create the element mesh: NOTE THIS DOES NOT INCLUDE THE FILTER !!!
    // find the geometric partitioning of the nodal mesh, so the element mesh will coincide 
    // with the nodal mesh
    PetscInt md,nd; 
    ierr = DMDAGetInfo(da_nodes,NULL,NULL,NULL,NULL,&md,&nd,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
    
    CHKERRQ(ierr);
    
    // vectors with partitioning information
    PetscInt *Lx=new PetscInt[md];
    PetscInt *Ly=new PetscInt[nd];
    
    
    // get number of nodes for each partition
    const PetscInt *LxCorrect, *LyCorrect ;
    ierr = DMDAGetOwnershipRanges(da_nodes, &LxCorrect, &LyCorrect, NULL); 
    
    
    
    
    CHKERRQ(ierr);
    
    // subtract one from the lower left corner.
    for (int i=0; i<md; i++){
        Lx[i] = LxCorrect[i];
        if (i==0){Lx[i] = Lx[i]-1;}
    }
    for (int i=0; i<nd; i++){
        Ly[i] = LyCorrect[i];
        if (i==0){Ly[i] = Ly[i]-1;}
    }
    
    // Create the element grid: NOTE CONNECTIVITY IS 0
    PetscInt conn = 0;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,nx-1,ny-1,md,nd,
                        1,conn,Lx,Ly,&(da_elem));
    CHKERRQ(ierr);
    
    // 	// Initialize
    //         DMSetFromOptions(da_elem);
    //         DMSetUp(da_elem);
    // 	
    // Set element center coordinates
    ierr = DMDASetUniformCoordinates(da_elem , xmin+dx/2.0,xmax-dx/2.0, ymin+dy/2.0,ymax-dy/2.0, 0.0,0.0);
    CHKERRQ(ierr);
    
    // Clean up
    delete [] Lx;
    delete [] Ly;
    
    return(ierr);
}

PetscErrorCode TopOpt::SetUpOPT(){
    
    PetscErrorCode ierr;
    
    //ierr = VecDuplicate(CRAPPY_VEC,&xPhys); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da_elem,&xPhys);  CHKERRQ(ierr);
    // Total number of design variables
    VecGetSize(xPhys,&n);
    
    PetscBool flg;
    
    // Optimization paramteres
    PetscOptionsGetReal(NULL,NULL,"-Emin",&Emin,&flg);
    PetscOptionsGetReal(NULL,NULL,"-Emax",&Emax,&flg);
    PetscOptionsGetReal(NULL,NULL,"-volfrac",&volfrac,&flg);
    PetscOptionsGetReal(NULL,NULL,"-penal",&penal,&flg);
    PetscOptionsGetReal(NULL,NULL,"-rmin",&rmin,&flg);
    PetscOptionsGetInt(NULL,NULL,"-maxItr",&maxItr,&flg);
    PetscOptionsGetInt(NULL,NULL,"-filter",&filter,&flg);
    PetscOptionsGetReal(NULL,NULL,"-Xmin",&Xmin,&flg);
    PetscOptionsGetReal(NULL,NULL,"-Xmax",&Xmax,&flg);
    PetscOptionsGetReal(NULL,NULL,"-movlim",&movlim,&flg);
    
    PetscPrintf(PETSC_COMM_WORLD,"################### Optimization settings ####################\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Problem size: n= %i, m= %i\n",n,m);
    PetscPrintf(PETSC_COMM_WORLD,"# -filter: %i  (0=sens., 1=dens., 2=PDE)\n",filter);
    PetscPrintf(PETSC_COMM_WORLD,"# -rmin: %f\n",rmin);
    PetscPrintf(PETSC_COMM_WORLD,"# -volfrac: %f\n",volfrac);
    PetscPrintf(PETSC_COMM_WORLD,"# -penal: %f\n",penal);
    PetscPrintf(PETSC_COMM_WORLD,"# -Emin/-Emax: %e - %e \n",Emin,Emax);
    PetscPrintf(PETSC_COMM_WORLD,"# -maxItr: %i\n",maxItr);
    PetscPrintf(PETSC_COMM_WORLD,"# -movlim: %f\n",movlim);
    PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
    
    // Allocate after input
    gx = new PetscScalar[m];
    if (filter==0){
        Xmin = 0.001; // Prevent division by zero in filter
    }
    
    // Allocate the optimization vectors
    ierr = VecDuplicate(xPhys,&x); CHKERRQ(ierr);
    
/*    
    // Seeding
    srand (static_cast <unsigned> (time(0)));

    PetscInt nn;
    VecGetSize(x,&nn);   
    for ( PetscInt i=0;i<nn;i++ ){
        float r = 0.2 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.5-0.2)));
        VecSetValue(x,i,r,INSERT_VALUES);
    }
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    
    ierr = VecCopy(x, xPhys);CHKERRQ(ierr);*/


    VecSet(x,volfrac); // Initialize to volfrac !
    VecSet(xPhys,volfrac); // Initialize to volfrac !

    
    
    // Sensitivity vectors
    ierr = VecDuplicate(x,&dfdx); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(x,m, &dgdx); CHKERRQ(ierr);
    
    // Bounds and 
    VecDuplicate(x,&xmin);
    VecDuplicate(x,&xmax);
    VecDuplicate(x,&xold);	
    VecSet(xold,volfrac);
    //ierr = VecCopy(x, xold);CHKERRQ(ierr);
    return(ierr);
}

PetscErrorCode TopOpt::PeriodizationMatrix(TopOpt *opt){
    
    PetscErrorCode ierr;
    
    PetscInt Gsize;
    PetscInt NodeID, NodeID0, NodeID1, NodeIDX, NodeIDY, NodeIDXY;
    PetscInt nn;
    PetscInt Indx0, Indy0;
    PetscScalar *lcoorp;
    Vec lcoor;
    
    // Get  global index
    ierr = VecGetSize(x,&Gsize);CHKERRQ(ierr);
    // Create Mapping Matrix DT
    MatCreateAIJ(PETSC_COMM_WORLD , PETSC_DETERMINE, PETSC_DETERMINE, Gsize, Gsize, divx*divy*8, NULL,divx*divy*8, NULL, &DT);
    
    // Create Local to Global mapping and setting the mapping matrix 
    ISLocalToGlobalMapping tmapping;
    DMGetLocalToGlobalMapping(da_elem, &tmapping);
    MatSetLocalToGlobalMapping(DT,tmapping,tmapping);
    
    // Get Global Indices
    DMGetLocalToGlobalMapping(da_elem, &tmapping);
    const PetscInt *gidx;
    ISLocalToGlobalMappingGetIndices(tmapping,&gidx);
    
    // Create Index Maps from Standard to PETsc and reverse
    PetscInt *IndexMapMTP = new PetscInt[Gsize]; 
    PetscInt *IndexMapPTM = new PetscInt[Gsize]; 
    
    // Allocate Index Maps to zero
    memset(IndexMapMTP, 0, sizeof(IndexMapMTP[0])*Gsize);
    memset(IndexMapPTM, 0, sizeof(IndexMapPTM[0])*Gsize);
    
    // Get Local Coordinates
    ierr = DMGetCoordinatesLocal(da_elem,&lcoor); CHKERRQ(ierr);
    VecGetArray(lcoor,&lcoorp); // Putting local coordinates into pointer
    
    VecGetLocalSize(opt->x,&nn);
    
    // Set up Mapping index:
    for (PetscInt i=0;i<nn;i++){
        
        PetscInt j = 2*i;
        Indx0 = floor((lcoorp[j])/dx);
        Indy0 = floor((lcoorp[j+1])/dy);
        NodeID0 = Indy0*(nxyz[0]-1)+Indx0;
        IndexMapMTP[NodeID0] = gidx[i];
        IndexMapPTM[gidx[i]] = NodeID0;
        
    }
    // Clear usage of lcoorp
    VecRestoreArray(lcoor,&lcoorp);
    
    // gather global dofs in array and divide with IndexMaxCOUNT
    PetscInt *tmp= new PetscInt[Gsize];
    
    memset(tmp,0, sizeof(tmp[0])*Gsize);
    
    MPI_Allreduce(IndexMapMTP,tmp,Gsize,MPIU_INT, MPI_MAX,PETSC_COMM_WORLD );
    memcpy(IndexMapMTP,tmp,sizeof(PetscInt)*Gsize);    
    
    memset(tmp,0, sizeof(tmp[0])*Gsize);
    
    MPI_Allreduce(IndexMapPTM,tmp,Gsize,MPIU_INT, MPI_MAX,PETSC_COMM_WORLD );
    memcpy(IndexMapPTM,tmp,sizeof(PetscInt)*Gsize);    
    
    delete [] tmp;   
    
    // Set up the mapping matrix DT  
    for (PetscInt i=0;i<nn;i++){
        
        // Find node ID's in standard representation
        NodeID = IndexMapPTM[gidx[i]];
        PetscInt Indx[divx];
        PetscInt Indsymx[divx];
        PetscInt Indy[divy];
        PetscInt Indsymy[divy];
        PetscInt NodeD1, NodeD2, NodeD3, NodeD4;
        //Set indices in x
        for (PetscInt ii=0;ii<divx;ii++){
            Indx[ii] = ( NodeID%(nxyz[0]-1) ) % ( (nxyz[0]-1)/divx ) + ii*(nxyz[0]-1)/divx;
            Indsymx[ii] = (nxyz[0]-1)/divy-Indx[ii]%((nxyz[0]-1)/divy) + ii*((nxyz[0]-1)/divy)-1 ;
            
        }
        
        //Set indices in y
        for (PetscInt ii=0;ii<divy;ii++){
            Indy[ii] = PetscInt(floor(NodeID/(nxyz[0]-1.0)))%((nxyz[1]-1)/divy) + ii*(nxyz[1]-1)/divy;
            Indsymy[ii] = (nxyz[1]-1)/divy-Indy[ii]%((nxyz[1]-1)/divy) + ii*((nxyz[1]-1)/divy)-1;
        }
        
        for (PetscInt ii=0;ii<divx;ii++){
            for (PetscInt ij=0;ij<divy;ij++){
                NodeID1 = Indy[ij]*(nxyz[1]-1) + Indx[ii];
                NodeIDX = Indsymy[ij]*(nxyz[1]-1) + Indx[ii];
                NodeIDY = Indy[ij]*(nxyz[1]-1) + Indsymx[ii];
                NodeIDXY = Indsymy[ij]*(nxyz[1]-1) + Indsymx[ii];
            
                // DIAGONALIZE
                NodeD1 = Indx[ij]*(nxyz[1]-1) + Indy[ii];
                NodeD2 = Indsymx[ij]*(nxyz[1]-1) + Indy[ii];
                NodeD3 = Indx[ij]*(nxyz[1]-1) + Indsymy[ii];
                NodeD4 = Indsymx[ij]*(nxyz[1]-1) + Indsymy[ii];
                // PetscPrintf(PETSC_COMM_SELF,"%i %i %i %i %i %i %i %i\n",NodeID1,NodeIDX,NodeIDY,NodeIDXY,NodeD1,NodeD2,NodeD3,NodeD4);
                
                // PetscPrintf(PETSC_COMM_SELF,"Petsc: %i Our: %i\n",IndexMapMTP[NodeID1],NodeID1);
                //
                if ( MapType == 1 ) { // Unit cell mapping
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeID1],1.0/(divx*divy),INSERT_VALUES); 
                }
                else if ( MapType == 2 ) { // Unit cell mapping with x-symmetry
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeID1],1.0/(divx*divy*2),INSERT_VALUES); 
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDX],1.0/(divx*divy*2),INSERT_VALUES);
                }
                else if ( MapType == 3 ) { // Unit cell mapping with y-symmetry
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeID1],1.0/(divx*divy*2),INSERT_VALUES); 
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDY],1.0/(divx*divy*2),INSERT_VALUES);    
                }
                else if ( MapType == 4 ) { // Unit cell mapping with x- and y-symmetry
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeID1],1.0/(divx*divy*4),INSERT_VALUES);                
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDX],1.0/(divx*divy*4),INSERT_VALUES);
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDY],1.0/(divx*divy*4),INSERT_VALUES);
                    MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDXY],1.0/(divx*divy*4),INSERT_VALUES);
                }
                else if ( MapType == 5 ) { // Unit cell mapping with 8 diagonal elements
                    // DIAGONALIZE
                    if ( NodeID1 == NodeD1 || NodeID1 == NodeD4 ) {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD1],1.0/(divx*divy*4),INSERT_VALUES);
                    } else {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD1],1.0/(divx*divy*8),INSERT_VALUES); 
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeID1],1.0/(divx*divy*8),INSERT_VALUES); 
                    }
                    if ( NodeIDX == NodeD2 || NodeIDX == NodeD3 ) {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD2],1.0/(divx*divy*4),INSERT_VALUES);
                    } else {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD2],1.0/(divx*divy*8),INSERT_VALUES);  
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDX],1.0/(divx*divy*8),INSERT_VALUES);
                    }
                    if ( NodeIDY == NodeD3 || NodeIDY == NodeD2 ) {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD3],1.0/(divx*divy*4),INSERT_VALUES);  
                    } else {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD3],1.0/(divx*divy*8),INSERT_VALUES); 
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDY],1.0/(divx*divy*8),INSERT_VALUES);   
                    }
                    if ( NodeIDXY == NodeD4 || NodeIDXY == NodeD1  ) {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD4],1.0/(divx*divy*4),INSERT_VALUES);   
                    } else {
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeD4],1.0/(divx*divy*8),INSERT_VALUES); 
                        MatSetValue(DT,gidx[i],IndexMapMTP[NodeIDXY],1.0/(divx*divy*8),INSERT_VALUES); 
                    }
                }
            }
        }
    } 
    
    // Assemble DT across processors
    MatAssemblyBegin(DT,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(DT,MAT_FINAL_ASSEMBLY);
    return ierr;
}

PetscErrorCode TopOpt::DMDAGetElements_2D(DM dm,PetscInt *nel,PetscInt *nen,const PetscInt *e[]) {
    PetscErrorCode ierr;
    DM_DA          *da = (DM_DA*)dm->data;
    PetscInt       i,xs,xe,Xs,Xe;
    PetscInt       j,ys,ye,Ys,Ye;
    PetscInt       cnt=0, cell[4], ns=1, nn=4;
    PetscInt       c;
    if (!da->e) {
        if (da->elementtype == DMDA_ELEMENT_Q1) {ns=1; nn=4;}
        ierr = DMDAGetCorners(dm,&xs,&ys,NULL,&xe,&ye,NULL);CHKERRQ(ierr);
        ierr = DMDAGetGhostCorners(dm,&Xs,&Ys,NULL,&Xe,&Ye,NULL);CHKERRQ(ierr
        );
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

PetscErrorCode TopOpt::AllocateMMAwithRestart(PetscInt *itr, MMA **mma)  {
    
    PetscErrorCode ierr = 0;
    
    // Set MMA parameters (for multiple load cases)
    PetscScalar aMMA[m];
    PetscScalar cMMA[m];
    PetscScalar dMMA[m];
    for (PetscInt i=0;i<m;i++){
        aMMA[i]=0.0;
        dMMA[i]=0.0;
        cMMA[i]=1000.0;
    }
    
    // Check if restart is desired
    restart = PETSC_TRUE; // DEFAULT USES RESTART
    flip = PETSC_TRUE;     // BOOL to ensure that two dump streams are kept
    PetscBool onlyLoadDesign = PETSC_FALSE; // Default restarts everything
    
    // Get inputs
    PetscBool flg;
    char filenameChar[PETSC_MAX_PATH_LEN];
    PetscOptionsGetBool(NULL,NULL,"-restart",&restart,&flg);
    PetscOptionsGetBool(NULL,NULL,"-onlyLoadDesign",&onlyLoadDesign,&flg);
    
    if (restart) {
        ierr = VecDuplicate(x,&xo1); CHKERRQ(ierr);
        ierr = VecDuplicate(x,&xo2); CHKERRQ(ierr);
        ierr = VecDuplicate(x,&U); CHKERRQ(ierr);
        ierr = VecDuplicate(x,&L); CHKERRQ(ierr);
    }
    
    // Determine the right place to write the new restart files
    std::string filenameWorkdir = "./";
    PetscOptionsGetString(NULL,NULL,"-workdir",filenameChar,sizeof(filenameChar),&flg);
    if (flg){
        filenameWorkdir = "";
        filenameWorkdir.append(filenameChar);
    }
    filename00 = filenameWorkdir;
    filename00Itr = filenameWorkdir;
    filename01 = filenameWorkdir;
    filename01Itr = filenameWorkdir;
    
    filename00.append("/Restart00.dat");
    filename00Itr.append("/Restart00_itr_f0.dat");
    filename01.append("/Restart01.dat");
    filename01Itr.append("/Restart01_itr_f0.dat");
    
    // Where to read the restart point from
    std::string restartFileVec = ""; // NO RESTART FILE !!!!!
    std::string restartFileItr = ""; // NO RESTART FILE !!!!!
    
    PetscOptionsGetString(NULL,NULL,"-restartFileVec",filenameChar,sizeof(filenameChar),&flg);
    if (flg) {
        restartFileVec.append(filenameChar);
    }
    PetscOptionsGetString(NULL,NULL,"-restartFileItr",filenameChar,sizeof(filenameChar),&flg);
    if (flg) {
        restartFileItr.append(filenameChar);
    }
    
    // Which solution to use for restarting
    PetscPrintf(PETSC_COMM_WORLD,"##############################################################\n");
    PetscPrintf(PETSC_COMM_WORLD,"# Continue from previous iteration (-restart): %i \n",restart);
    PetscPrintf(PETSC_COMM_WORLD,"# Restart file (-restartFileVec): %s \n",restartFileVec.c_str());
    PetscPrintf(PETSC_COMM_WORLD,"# Restart file (-restartFileItr): %s \n",restartFileItr.c_str());
    PetscPrintf(PETSC_COMM_WORLD,"# New restart files are written to (-workdir): %s (Restart0x.dat and Restart0x_itr_f0.dat) \n",filenameWorkdir.c_str());
    
    // Check if files exist:
    PetscBool vecFile = fexists(restartFileVec);
    if (!vecFile) { PetscPrintf(PETSC_COMM_WORLD,"File: %s NOT FOUND \n",restartFileVec.c_str()); }
    PetscBool itrFile = fexists(restartFileItr);
    if (!itrFile) { PetscPrintf(PETSC_COMM_WORLD,"File: %s NOT FOUND \n",restartFileItr.c_str()); }
    
    // Read from restart point
    
    PetscInt nGlobalDesignVar;
    VecGetSize(x,&nGlobalDesignVar); // ASSUMES THAT SIZE IS ALWAYS MATCHED TO CURRENT MESH
    if (restart && vecFile && itrFile){
        
        PetscViewer view;
        // Open the data files 
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,restartFileVec.c_str(),FILE_MODE_READ,&view);	
        
        VecLoad(x,view);
        VecLoad(xPhys,view);
        VecLoad(xo1,view);
        VecLoad(xo2,view);
        VecLoad(U,view);
        VecLoad(L,view);
        PetscViewerDestroy(&view);
        
        // Read iteration and fscale
        std::fstream itrfile(restartFileItr.c_str(), std::ios_base::in);
        itrfile >> itr[0];
        itrfile >> fscale;
        
        
        // Choose if restart is full or just an initial design guess
        if (onlyLoadDesign){
            PetscPrintf(PETSC_COMM_WORLD,"# Loading design from file: %s \n",restartFileVec.c_str());
            *mma = new MMA(nGlobalDesignVar,m,x, aMMA, cMMA, dMMA);
        }
        else {
            PetscPrintf(PETSC_COMM_WORLD,"# Continue optimization from file: %s \n",restartFileVec.c_str());
            *mma = new MMA(nGlobalDesignVar,m,*itr,xo1,xo2,U,L,aMMA,cMMA,dMMA);
        }
        
        PetscPrintf(PETSC_COMM_WORLD,"# Successful restart from file: %s and %s \n",restartFileVec.c_str(),restartFileItr.c_str());
    }
    else {
        *mma = new MMA(nGlobalDesignVar,m,x,aMMA,cMMA,dMMA);
    }  
    
    return ierr;
} 


PetscErrorCode TopOpt::WriteRestartFiles(PetscInt *itr, MMA *mma) {
    
    PetscErrorCode ierr=0;
    // Only dump data if correct allocater has been used
    if (!restart){
        return -1;
    }
    
    // Get restart vectors
    mma->Restart(xo1,xo2,U,L);
    
    // Choose previous set of restart files
    if (flip){ flip = PETSC_FALSE; 	}	
    else {     flip = PETSC_TRUE; 	}
    
    // Write file with iteration number of f0 scaling
    // and a file with the MMA-required vectors, in the following order:
    // : x,xPhys,xold1,xold2,U,L
    PetscViewer view; // vectors
    PetscViewer restartItrF0; // scalars
    
    PetscViewerCreate(PETSC_COMM_WORLD, &restartItrF0);
    PetscViewerSetType(restartItrF0, PETSCVIEWERASCII);
    PetscViewerFileSetMode(restartItrF0, FILE_MODE_WRITE);
    
    // Open viewers for writing
    if (!flip){
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename00.c_str(),FILE_MODE_WRITE,&view);
        PetscViewerFileSetName(restartItrF0, filename00Itr.c_str());
    }
    else if (flip){
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename01.c_str(),FILE_MODE_WRITE,&view);
        PetscViewerFileSetName(restartItrF0, filename01Itr.c_str());
    }
    
    // Write iteration and fscale
    PetscViewerASCIIPrintf(restartItrF0, "%d ", itr[0]);
    PetscViewerASCIIPrintf(restartItrF0," %e",fscale);
    PetscViewerASCIIPrintf(restartItrF0,"\n");
    
    // Write vectors
    VecView(x,view); // the design variables
    VecView(xPhys,view);
    VecView(xo1,view);
    VecView(xo2,view);
    VecView(U,view);	
    VecView(L,view);
    
    // Clean up
    PetscViewerDestroy(&view);
    PetscViewerDestroy(&restartItrF0);
    
    //PetscPrintf(PETSC_COMM_WORLD,"DONE WRITING DATA\n");
    return ierr;
}
