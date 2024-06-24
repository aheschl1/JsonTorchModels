import torch
import torch.nn as nn

from json_torch_models.utils import my_import


class JsonPyTorchModel(nn.Module):

    def __init__(self, children: list, tag=None) -> None:
        """
        Builds a module based on a list of children.
        :param children: List of children in dictionary form.
        """
        super(JsonPyTorchModel, self).__init__()
        self.tag = tag
        self.child_modules = children
        self.data = {}
        self.skipped_connection_info = {}
        self.network_modules = nn.ModuleList([])
        self._construct()

    def _construct(self) -> None:
        """
        Constructs the internal module based on children list.
        :return: None
        """
        for i, child in enumerate(self.child_modules):
            if 'children' in child.keys():
                self.network_modules.append(JsonPyTorchModel(child['children'], tag=child.get('tag', None)))
                return
            self.network_modules.append(
                my_import(child['component_class'])(**(child['args']))
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

                self.skipped_connection_info[i] = this_operation

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass and manages skipped connections.
        :param x: The data to compute.
        :return: The output data.
        """
        for i, module in enumerate(self.network_modules):
            if not isinstance(x, tuple):
                x = (x,)

            skipped_operation = self.skipped_connection_info.get(i, {})

            if "forward_in" not in skipped_operation:
                x = module(*x)
            else:
                # Replace the map of "key" : "variable" with "key" : value
                forward_in = {key: self.data[value] for key, value in skipped_operation['forward_in'].items()}
                x = module(*x, forward_in)

            if 'store_out' in skipped_operation:
                self.data[skipped_operation['store_out']] = x

        return x
