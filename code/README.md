# Getting started

There are a few steps to take to run the code for this lab on your machine.


## Creating a virtual environment

First, it is important to isolate the code and the packages it depends upon from those of other projects on your machine. Therefore we strongly encourage you to use a virtual environment.

Using Python version 3.3 or above, you can create a virtual environment with:

```
$ python -m venv .venv
```

This command will create an empty virtual environment named `.venv` in the current directory. After creation, you can activate the virtual environment with:

```
$ source .venv/bin/activate
```

This environment has its own independent set of packages, installed in the `.venv` directory, and is isolated from the base environment and other virtual environments.

You can deactivate the virtual environment with:

```
$ deactivate
```

For further information, read the [venv documentation](https://docs.python.org/3/library/venv.html).


### Installing Python packages in the environment

Next, you will need to install the following packages in the newly created virtual environment to run the code for this lab.

- torch
- torchvision
- matplotlib
- numpy

Use the following command to install these dependencies:

```
$ pip install torch torchvision matplotlib numpy
```

__Important: Make sure the virtual environment is activated before installing these packages!__

If you would like to install PyTorch with GPU hardware acceleration enabled, consult the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the most up-to-date information.


## Working with Jupyter Notebooks

Finally, you will need to install the IPython kernel and configure your IDE to work with Jupyter Notebooks.

Make sure the virtual environment is activated. Then install the `ipykernel` package with:

```
$ pip install ipykernel
```

PyCharm Community Edition does not support Jupyter Notebooks. However, you can run them locally in the browser using the `jupyter` package. Install this package with:

```
$ pip install jupyter
```

Next, run Jupyter in your browser by running the following command in the terminal. This will open a browser-based application where you can open, edit, and run the Jupyter Notebook files (.ipynb).

```
$ jupyter notebook
```

Visual Studio Code has native support for working with Jupyter Notebooks. Follow the instructions in [Manage Jupyter Kernels in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) to select the kernel from the virtual environment. Editing and running notebooks is explained in [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

For further information on the ins and outs of Jupyter Notebooks, explore the [Jupyter website](https://jupyter.org/) or read the [IPython kernel documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).
