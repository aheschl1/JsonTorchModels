import json
from typing import Union, List

import torch

from json_torch_models.model import JsonPyTorchModel
from json_torch_models.utils import my_import, PackageLookupBin
from torch import nn


class ModelFactory:
    def __init__(self, json_path: str, lookup_packages: List[str] = None) -> None:
        """
        Given a path to a json model skeleton, helps builds a model, and verifies that the json is correct.
        :param json_path: The path to the json file to parse.
        :param lookup_packages: extra packages in which to look for modules.
        """
        if lookup_packages is not None:
            PackageLookupBin.lookup_paths.extend(lookup_packages)
        self.json_path = json_path
        self.model = None
        self.log_kwargs = None
        self._build_architecture()

    def _build_architecture(self) -> None:
        """
        Builds the model and stores it in self model variable.
        :return: Nothing
        """
        with open(self.json_path, "r") as file:
            model_definition = json.load(file)

        self.log_kwargs = model_definition.get('LogKwargs', None)

        if 'NoWrap' in model_definition.keys():
            # This is to be used if you just want to point to a pre-writen network
            self.model = my_import(model_definition['Only']['component_class'])
            return

        is_unet = any([key in model_definition for key in ['Encoder', 'Decoder', 'Middle']])
        verifier = ModelFactory._verify_unet_structure if is_unet else ModelFactory._verify_structure
        verifier(model_definition)

        if is_unet:
            model_definition = ModelFactory._convert_unet_like_to_normal(model_definition)

        model = JsonPyTorchModel(model_definition['children'], tag=model_definition.get('tag', None))

        self.model = model

    def get_model(self) -> nn.Module:
        """
        Returns the generated model.
        """
        return self.model

    def get_log_kwargs(self) -> Union[dict, None]:
        """
        Returns the log args that were specified in the json.
        :return:
        """
        return self.log_kwargs

    @staticmethod
    def _convert_unet_like_to_normal(model_definition: dict) -> dict:
        """
        Convert a json that was defined with unet syntax into the normal fully sequential representation.
        :param model_definition: The model dictionary.
        :return: The updated model dictionary.
        """
        encoder_elements = model_definition['Encoder']
        middle_elements = model_definition['Middle']
        decoder_elements = model_definition['Decoder']

        sequential = []
        sequential += encoder_elements
        sequential += middle_elements
        sequential += decoder_elements

        new_root = {
            "tag": "Root",
            "children": sequential
        }

        return new_root

    @staticmethod
    def _verify_structure(model_definition: dict) -> bool:
        """
        Verifies that the structure of a json model is valid.
        :param model_definition: The model dictionary to verify.
        :return: Returns true if valid, raises an exception otherwise.
        """
        keys = model_definition.keys()

        ModelFactory._validate_node(keys)

        if 'tag' in keys:
            if not isinstance(model_definition['tag'], str):
                raise InvalidJsonArchitectureException("'tag' must be an instance of 'str'")

        if 'children' in keys:
            if not isinstance(model_definition['children'], list):
                raise InvalidJsonArchitectureException("'children' must be an instance of 'list'")

        if 'component_class' in keys:
            if not isinstance(model_definition['component_class'], str):
                raise InvalidJsonArchitectureException("'component_class' must be an instance of 'str'")

        if 'args' in keys:
            if not isinstance(model_definition['args'], dict):
                raise InvalidJsonArchitectureException("'args' must be an instance of 'dict'")

        if 'store_out' in keys:
            if not isinstance(model_definition['store_out'], str):
                raise InvalidJsonArchitectureException("'store_out' must be an instance of 'str'")

        if 'children' not in keys and 'component_class' not in keys:
            raise InvalidJsonArchitectureException("You defined an unmeaningful node.: " + str(model_definition))

        if 'children' in keys:
            for child in list(model_definition['children']):
                ModelFactory._verify_structure(child)

        return True

    @staticmethod
    def _verify_unet_structure(model_definition: dict) -> bool:
        """
        Verifies that the structure of a json model is valid.
        :param model_definition: The model dictionary to verify.
        :return: Returns true if valid, raises an exception otherwise.
        """
        keys = model_definition.keys()
        if any([elem not in keys for elem in ['Encoder', 'Decoder', 'Middle']]):
            raise InvalidJsonArchitectureException("When defining a UNet like structure you must define: Encoder, Decoder, and Middle")
        if not (
                isinstance(model_definition['Encoder'], list) and
                isinstance(model_definition['Encoder'], list) and
                isinstance(model_definition['Encoder'], list)):
            raise InvalidJsonArchitectureException("When defining a UNet like structure (you defined 'Encoder')" +
                                                   " the portions should be defined as lists!")

        for element in model_definition['Encoder']:
            ModelFactory._verify_structure(element)
        for element in model_definition['Decoder']:
            ModelFactory._verify_structure(element)
        for element in model_definition['Middle']:
            ModelFactory._verify_structure(element)

        return True

    @staticmethod
    def _validate_node(keys) -> bool:
        """
        Verifies that the keys of a node are valid.
        :param keys: The list of keys.
        :return: Returns true if all keys are valid, raises an exception otherwise.
        """
        if 'tag' in keys and 'children' not in keys:
            raise InvalidJsonArchitectureException("'tag' is only defined in 'Module' nodes. Without 'children' argument, this is invalid!.")
        if 'tag' in keys and 'args' in keys:
            raise InvalidJsonArchitectureException("'tag' is only defined in 'Module' nodes. With 'args' argument, this is invalid!.")
        if 'component_class' in keys and 'args' not in keys:
            raise InvalidJsonArchitectureException("You must define 'component_class' and 'args' together in a 'Component' node.")
        return True


class InvalidJsonArchitectureException(Exception):
    """
    Exception raised when a user tries to build a model with invalid json.
    """

    def __init__(self, message="Invalid model architecture in json."):
        super().__init__(message)


if __name__ == "__main__":
    factory = ModelFactory("../examples/pytorch_efficient_net_skipped.json")
    model = factory.get_model()
    print(model)
    x = torch.randn(5, 1, 224, 224)
    y = model(x)
    print(x.shape, y.shape)
    print(factory.get_log_kwargs())
    print("Done.")
