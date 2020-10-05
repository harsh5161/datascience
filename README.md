# protonautoml_webapp_v1

In order to set up a working development environment for working on protonAutoML products follow the following steps. <br />
Option 1: Go to the Dependencies Folder and open the req.txt file and install all of the mentioned Python and R packages along IPython-kernel/or Jupyter Notebook to your development environment created using Anaconda. <br />
Option 2: Download and Install Anaconda onto your system. Activate your Anaconda (base) environment using `conda activate` and then execute `conda env create --file env_name.yaml` <br />

If you are using option 2; <br />
For a MacOS 10.15(Catalina) environment replace env_name.yaml with macos_env.yaml [The file can be found in Dependencies Folder] <br />
For a Windows10 environment replace env_name.yaml with windows_env.yaml  [The file can be found in Dependencies Folder] <br />
##Note## <br />
In the respective files, there's a part called "prefix": please replace it with your respective anaconda PATH where you want the environment to be created <br />
