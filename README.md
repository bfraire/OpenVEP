# SimpleVEP

## Setup for Windows 11
```
pip install virtualenv
virtualenv pyenv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pyenv\Scripts\activate
(Install "Desktop development with C++" workload: https://visualstudio.microsoft.com/visual-cpp-build-tools/)
pip install -r requirements.txt
```
All of the Python packages would be installed into the folder `pyenv/`, so they can be easily removed by simply deleting the folder. When you need to install additional packages, you can activate the virtual environment by running `pyenv\Scripts\activate` and then install the packages using `pip install`. If you want to deactivate the virtual environment, you can run `deactivate`.