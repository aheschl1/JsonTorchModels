import torch
import torch.nn as nn

from json_torch_models.utils import my_import


class JsonPyTorchModel(nn.Module):

    def __init__(self, tag: str, children: list) -> None:
        """
        Builds a module based on a list of children.
        :param tag: Name for the current module. Does nothing.
        :param children: List of children in dictionary form.
        """
        super(JsonPyTorchModel, self).__init__()
        self.tag = tag
        self.child_modules = children
        self.data = {}
        self.skipped_connections = {}
        self.network_modules = nn.ModuleList([])
        self._construct()

    def _construct(self) -> None:
        """
        Constructs the internal module based on children list.
        :return: None
        """
        for child in self.child_modules:
            if 'Tag' in child.keys():
                self.network_modules.append(JsonPyTorchModel(
                    child['Tag'],
                    child['Children']
                ))
                return

            self.network_modules.append(
                my_import(child['ComponentClass'])(**(child['args']))
            )

            if 'store_out' in child.keys() or 'forward_in' in child.keys():
                # New operation
                this_operation = {}
                if 'store_out' in child.keys():
                    this_operation['store_out'] = child['store_out']
                if 'forward_in' in child.keys():
                    if not isinstance(child['forward_in'], dict):
                        child['forward_in'] = {
                            child['forward_in']: child['forward_in']
                        }
                    this_operation['forward_in'] = child['forward_in']

                self.skipped_connections[len(self.skipped_connections)] = this_operation

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass and manages skipped connections.
        :param x: The data to compute.
        :return: The output data.
        """
        for i, module in enumerate(self.network_modules):
            skipped_operation = self.skipped_connections.get(i, {})

            if "forward_in" not in skipped_operation:
                x = module(*x)
            else:
                # Replace the map of "key" : "variable" with "key" : value
                forward_in = {key: self.data[value] for key, value in skipped_operation['forward_in'].items()}
                x = module(*x, forward_in)

            if 'store_out' in skipped_operation:
                self.data[skipped_operation['store_out']] = x

        return x
