#include <WriteVTS.h>

WriteVTS::WriteVTS(TopOpt *opt){
    
    PetscOptionsGetString(NULL,NULL,"-workdir",filenameChar,sizeof(filenameChar),&flg);
}

WriteVTS::~WriteVTS(){

}

PetscErrorCode WriteVTS::WritetoVTS(TopOpt *opt, PetscInt itr){
        
    PetscErrorCode ierr;
    
    std::stringstream ss;
    std::string teststr = "/Des";
    ss<<teststr<<"_"<<itr<<".vts ";
    ss>>teststr;	
    
    std::string teststrI = "./Des";
    ss<<teststrI<<"_"<<itr<<".vts ";
    ss>>teststrI;	
    
    if (flg){
        filenameI="";
        filenameI.append(filenameChar);
        filenameI.append(teststr);
    }
    else{
        filenameI="";
        filenameI.append(teststrI);
    }
    PetscViewer viewer; 
    ierr=PetscViewerVTKOpen(PETSC_COMM_WORLD,filenameI.c_str(),FILE_MODE_WRITE,&viewer);  CHKERRQ(ierr);
    ierr=VecView(opt->x,viewer); CHKERRQ(ierr);
    ierr=PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    
    
    teststr = "/ProDes";
    ss<<teststr<<"_"<<itr<<".vts ";
    ss>>teststr;	
    
    teststrI = "./ProDes";
    ss<<teststrI<<"_"<<itr<<".vts ";
    ss>>teststrI;	
    
    
    if (flg){
        filenameI="";
        filenameI.append(filenameChar);
        filenameI.append(teststr);
    }
    else{
        filenameI="";
        filenameI.append(teststrI);
    }
    
    ierr=PetscViewerVTKOpen(PETSC_COMM_WORLD,filenameI.c_str(),FILE_MODE_WRITE,&viewer);  CHKERRQ(ierr);
    ierr=VecView(opt->xPhys,viewer); CHKERRQ(ierr);
    ierr=PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    
    return ierr;
    
}