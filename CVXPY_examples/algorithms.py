import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any
torch.set_default_dtype(torch.double)

def fit(loss, params, X, Y, Xval, Yval, opt = torch.optim.Adam, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None):
    """
    Description:
    The function aims to minimize the loss of :math:`X` and :math:`Y` by updating the parameters using the gradient descent algorithm.

    Args:
        - loss (required):
            A criterion that measures the difference between :math:`X` and target :math:`Y`. The output is a scalar by default for backpropagation. 
        - params (``List[Tensor]``, required): 
            A list of parameters to optimize. 
        - X and Y (``Tensor``, required):
            Input and output tensors of training dataset.
        - Xval and Yval (``Tensor``, required):
            Input and output tensors of validation dataset.
        - opt (``Pytorch optimizer``, optional): 
            Pytorch optimizer is an optimization technique for gradient descent. Default: torch.optim.Adam.
        - opt_kwargs (``Dict[str, any]``, optional):
            A dict containing default values of optimization options. The default value for learning rate (float) is :math:`10^{-3}`.
        - batch_size (``int``, optional): 
            Training mini-batch size. Default: 128.
        - epochs (``int``, optional): 
            Training epochs. Default: 100.
        - verbose (``bool``, optional): 
            If set to True, display the training loss for each epoch. Default: False.
        - callback (``PyTorch callback``, optional): 
            PyTorch callback is a function triggered at specific points during model training. Default: None.

    Shape:
        - X: :math:`(*, )` where :math:`*` means any number of dimensions.
        - Y: :math:`(*, )` where the first dimension is the same shape as X.
        - Xval: :math:`(**, )` where :math:`**` means any number of dimensions.
        - Yval: :math:`(**, )` where the first dimension is the same shape as Xval.

    Returns:
        A list of validation loss for each epoch, a list of training loss for each epoch.

    Return type:
        ``List[loss scalar]``, ``List[loss scalar]``.

    Raises:
        ExceptionName: Description of the error raised, if any.

    Examples:
        >>> val_losses, train_losses = example_function(loss, params, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None)
    """

    train_dset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    opt = opt(params, **opt_kwargs)

    train_losses = []
    batch_loss = []
    val_losses = []
    try:
        for epoch in range(epochs):
            with torch.no_grad():
                val_losses.append(loss(Xval, Yval).item())
            if verbose:
                print("%03d | %3.5f" % (epoch + 1, val_losses[-1]))
            batch_loss = []
            for Xbatch, Ybatch in train_loader:
                opt.zero_grad()
                l = loss(Xbatch, Ybatch)
                l.backward()
                opt.step()
                batch_loss.append(l.item())
                if callback is not None:
                    callback()
            train_losses.append(np.mean(batch_loss))
            if verbose:
                print("epoch %03d | %3.5f" % (epoch, np.mean(batch_loss)))
    except Exception as e:
        print(f"Training stopped due to an error: {e}")
    finally:
        # Ensure valid_loss is returned even if training stops
        return val_losses, train_losses
