from collections.abc import Sequence
from typing import Optional

import torch
from aframe.architectures.zoo import Architecture
from aframe.architectures.autoencoder.skip_connection import SkipConnection


class Autoencoder(Architecture):
    def __init__(self, skip_connection: Optional[SkipConnection] = None):
        super().__init__()
        self.skip_connection = skip_connection
        self.blocks = torch.nn.ModuleList()

    def encode(self, *X: torch.Tensor, return_states: bool = False):
        states = []
        for block in self.blocks:
            if isinstance(X, tuple):
                X = block.encode(*X)
            else:
                X = block.encode(X)
            states.append(X)

        if return_states:
            return X, states[:-1]
        return X

    def decode(self, *X, states: Optional[Sequence[torch.Tensor]] = None):
        if self.skip_connection is not None and states is None:
            raise ValueError(
                "Must pass intermediate states when autoencoder "
                "has a skip connection function specified"
            )
        elif states is not None:
            if len(states) != len(self.blocks) - 1:
                raise ValueError(
                    "Passed {} intermediate states, expected {}".format(
                        len(states), len(self.blocks) - 1
                    )
                )
            states = [None] + states

        for i, block in enumerate(self.blocks[::-1]):
            if isinstance(X, tuple):
                X = block.decode(*X)
            else:
                X = block.decode(X)

            state = states[-i - 1]
            if state is not None:
                X = self.skip_connection(X, state)
        return X

    def forward(self, *X):
        return_states = self.skip_connection is not None
        X = self.encode(*X, return_states=return_states)
        if return_states:
            X, states = X
        else:
            states = None

        if isinstance(X, tuple):
            return self.decode(*X, states=states)
        return self.decode(X, states=states)
