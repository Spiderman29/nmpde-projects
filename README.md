### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

### Compiling instructions
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ ./main
```

### Downloading the mesh to run the program

1. Create a folder called `mesh` inside the `EspositoGrassiVenezia` directory.  
2. Download the test mesh from [this link](https://www.dropbox.com/scl/fi/a3099q5ulovb4hbfvs08w/mesh.msh?rlkey=0s0f0gi62pr7h8dq897zris73&st=826vvc5o&dl=0) and paste it into the `mesh/` folder.  
3. Modify the `parameters.prm` file inside the `params` directory with the name of the mesh you want to use.  

⚠️ *Note:* This is just a test mesh, since we are not authorized to publicly share the real mesh provided by the author of the article.


