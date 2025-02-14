This job was done by Théodore DE LEUSSE as part of his internship in the SciNeurotechLab supervised by Marco BONIZZATO.

This file allow us to discover how to use the differents models and to do some comparaison between them. 
What's more, the `A` part shows the quadratic complexity of the method using the Schur complement inversion, compared to the cubic complexity of the commonly used cholesky inverson. 

# GPcustom

A custom implementation of Gaussian Process models using different backends (GPytorch, NumPy).

- `FixedGP` &  `FixedOnlineGP` use only Numpy. They help us to understand what exactly is done in libraries such as gpytorch.
- `BOtorchModel` uses the botorch lib. I implemented the class but I didn't use it. But it can be usefull if you want to use some acquisition function from botorch in continuus space.
- `GPytorchFixed` & `GPytorchModel` use gpytorch. This is the one I used the most, especially for simulation in ``GPBOsimulator``.



## Installation 

1. Navigate to the GPcustom folder where the `setup.py` file is located using the terminal.
2. Enter the following command in the terminal: `pip install -e .`.
    This will locally install the GPcustom library.
3. You can then use this library simply with, for example: `from GPcustom.models import GPytorchFixed, GPytorchModel`.

## Usage

### Importing the Library

To use the GPcustom library in your project, you need to import the necessary modules:

```python
from GPcustom.models import GPytorchFixed, GPytorchModel
import torch
```

### Example

Here is a simple example of how to use the GPcustom library:

```python
import torch
from GPcustom.models import GPytorchFixed
import gpytorch
import matplotlib.pyplot as plt
from botorch.utils.transforms import standardize


# Example usage
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Generate example training data
train_X = torch.linspace(0, 1, 15, dtype=torch.float64).view(-1, 1)
train_Y = (torch.sin(train_X * (2 * torch.pi)) + 0.1 * torch.randn_like(train_X))[:, 0]
train_Y = standardize(train_Y)

# Fixed hyperparameters
fixed_params = {
    'lengthscale': 0.2,
    'outputscale': 1.0,
    'noise': 0.1
}

# Instantiate the fixed GP model
model = GPytorchFixed(
    train_X, train_Y, likelihood, 
    kernel_type='Matern52',
    **fixed_params
)

print(model.get_hyperparameters())

# Generate test points
test_X = torch.linspace(0, 1, 200, dtype=torch.float64).view(-1, 1)

# Make predictions
pred_mean, pred_std = model.predict(test_X)

# Plotting results
plt.figure(figsize=(8, 6))
plt.plot(train_X.numpy(), train_Y.numpy(), 'k*', label='Training Data')
plt.plot(test_X.numpy(), pred_mean.numpy(), 'b', label='Mean Prediction')
plt.fill_between(
    test_X.squeeze().numpy(),
    (pred_mean - 1.96 * pred_std).numpy(),
    (pred_mean + 1.96 * pred_std).numpy(),
    color='blue', alpha=0.2, label='Confidence Interval'
)
plt.legend()
plt.title("Fixed Gaussian Process Regression")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show() 
```

Dependencies
The following dependencies are required to use the GPcustom library:

numpy
torch
gpytorch
botorch
These dependencies are listed in the setup.py file and will be installed automatically when you install the GPcustom library.

Contact
If you have any questions or issues, please contact me at [deleussetheodore@gmail.com].