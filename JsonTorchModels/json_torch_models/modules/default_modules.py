import torch.nn as nn
import torch
from json_torch_models.model import JsonPyTorchModel

ADD = "add"
CONCAT = "concat"


class SkippedLinker(nn.Module):

    def __init__(self, mode: str, module: dict) -> None:
        """
        Can concatenate or add input for skipped connections before passing to a module.
        Used for JSON model architecture.
        """
        super().__init__()
        assert mode in [ADD, CONCAT]
        self.mode = mode
        self.module = JsonPyTorchModel(module['Tag'], module['Children'])

    def forward(self, x: torch.Tensor, extra: torch.Tensor):
        extra = list(extra.values())
        assert len(extra) == 1, "Can only use Linker with a single extra input value."
        extra = extra[0]

        if self.mode == ADD:
            return self.module(x + extra)
        return self.module(torch.concat((x, extra)))
