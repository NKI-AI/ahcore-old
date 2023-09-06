# encoding: utf-8
from __future__ import annotations

from pprint import pformat
from typing import Any, Optional, Type

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle


class ExtractFeaturesHook:
    """
    Context manager to add hooks to the layers of a PyTorch model to extract feature maps.

    Parameters
    ----------
    model : torch.nn.Module
        The model to which the hooks will be added.
    layer_names : list[str]
        The names of the layers from which the feature maps will be extracted.

    Attributes
    ----------
    features : dict
        The feature maps extracted from the layers.

    Examples
    --------
    >>> model = MyModel()
    >>> layer_names = ["conv1", "conv2"]
    >>> with ExtractFeaturesHook(model, layer_names) as hook:
    ...     output = model(input_data)
    ...     conv1_features = hook.features["conv1"]
    ...     conv2_features = hook.features["conv2"]
    """

    def __init__(self, model: nn.Module, *, layer_names: list[str] | None):
        self.model = model
        self.layer_names = layer_names if layer_names is not None else []
        self.hooks: list[RemovableHandle] = []
        self.features: dict[str, torch.Tensor] = {}

    def save_output(
        self,
        layer_name: str,
        module: nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Hook function that saves the output of a layer.

        In this setting we are not using the `module` or `input` parameters. However, they are required to be
        present in the function signature to be used as a hook. For future reference, they can be used as follows:
        - You can use module if you want to save or analyze the weights or biases of the layer.
        - You can use input if you want to see what goes into the layer,
        or if you want to save or analyze the input data.

        Parameters
        ----------
        layer_name : str
            The name of the layer.
        module : torch.nn.Module
            The layer module.
        input : torch.Tensor
            The input to the layer.
        output : torch.Tensor
            The output of the layer.
        """
        self.features[layer_name] = output.detach()

    def __enter__(self):
        """
        Registers the hooks when entering the context.
        """
        if not hasattr(self, "layer_cache"):
            self.layer_cache = {name: module for name, module in self.model.named_modules()}

        for layer_name in self.layer_names:
            if layer_name in self.layer_cache:
                module = self.layer_cache[layer_name]
                hook = module.register_forward_hook(
                    lambda module, input, output, layer_name=layer_name: self.save_output(
                        layer_name, module, input, output
                    )
                )
                self.hooks.append(hook)
            else:
                raise ValueError(
                    f"No layer named {layer_name} in model. "
                    f"These are the available ones: {pformat(self.layer_cache)}."
                )
        return self

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """
        Removes the hooks when exiting the context.

        Parameters
        ----------
        type : Optional[Type[BaseException]]
            The type of exception that caused the context to be exited, if any.
        value : Optional[BaseException]
            The instance of exception that caused the context to be exited, if any.
        traceback : Optional[Any]
            A traceback object encapsulating the call stack at the point where the exception originally occurred, if any.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
