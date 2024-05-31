import importlib
import os.path
import pkgutil
from typing import Any


class PackageLookupBin:
    lookup_paths = []
    if not os.path.exists(os.path.expanduser("~/.jsontorchmodels.lookup")):
        open(os.path.expanduser("~/.jsontorchmodels.lookup"), "w+")

    with open(os.path.expanduser("~/.jsontorchmodels.lookup"), "r") as file:
        for line in file.readlines():
            lookup_paths.append(line.strip())


def import_from(from_package: str, class_name: str) -> Any:
    module = importlib.import_module(from_package)
    # Iterate through all modules in the package
    for loader, name, is_pkg in pkgutil.walk_packages(module.__path__):
        # Import module
        submodule = importlib.import_module(f"{from_package}.{name}")
        # Check if class_name exists in the module
        if hasattr(submodule, class_name):
            return getattr(submodule, class_name)

    # If class is not found in any submodule, raise ImportError
    raise ImportError(f"Class '{class_name}' not found in package '{from_package}'")


def my_import(class_name: str, dropout_package: str = 'torch.nn'):
    """
    Returns a class based on a string name.
    :param class_name: The name of the object being searched for.
    :param dropout_package: The package to look into of the class isn't in the if/else statements
    :return: Any object defined in this long if/else sequence.
    """

    if "." in class_name:
        package = '.'.join(class_name.split(".")[:-1])
        class_name = class_name.split(".")[-1]
        module = importlib.import_module(package)
        class_ = getattr(module, class_name)
        return class_

    try:
        module = importlib.import_module("torchvision.models")
        class_ = getattr(module, class_name)
        return class_
    except AttributeError:
        ...

    try:
        module = importlib.import_module("json_torch_models.modules.default_modules")
        class_ = getattr(module, class_name)
        return class_
    except AttributeError:
        ...

    for module in PackageLookupBin.lookup_paths:
        try:
            module = importlib.import_module(module)
            class_ = getattr(module, class_name)
            return class_
        except AttributeError:
            ...

    try:
        module = importlib.import_module(dropout_package)
        class_ = getattr(module, class_name)
        return class_
    except AttributeError:
        raise NotImplementedError(f'The requested module {class_name} has not been placed in default_modules, and is '
                                  f'not in torch.nn. If you want to add a custom module for use with this project'
                                  f', clone the repository https://github.com/aheschl1/JsonTorchModels and add your '
                                  f'class to json_torch_models.modules.default_modules.py')
