**Json Pytorch Models**
This is an experimentation software built with quick arhitecture iteration in mind.
By outlining your model in json, you can create many configurations and structures quickly, without needing to have many python classes.

**How to use**

*Option 1*

1. pip install json-torch-models

*Option 2*

1. Navigate to the root directory after cloning.
2. pip install -e .
   
Define your models in json (as described below)

From any project, build the model with:

```py
from json_torch_models.model_factory import ModelFactory
factory =  ModelFactory("path/to/json/model.json")
model = factory.get_model() # or factory.model
kwargs = factory.get_log_kwargs() # or factory.log_kwargs

# continue like any other pytorch flow
out = model(in)
```
**Linking with Custom Modules**

You have some options:
1. Clone this repository, and add your class to json_torch_models.modules.default_modules where it will be found by class name.
2. Add an if statement to utils.py.
3. Pass a list of packages (all the way to the file) to ModelFactory with the argument lookup_packages.
4. Modify _~/.jsontorchmodules.lookup_ and add a package on each line where this package should search for modules.
5. Instead of only giving the class name, you can give full.path.name.Component as Component Class

There are two levels of syntax to keep in mind.
1. Modules
3. Components
   
**Default Syntax**

*Modules*
```
{
  "LogKwargs": {"log": "stuff", "not": "required"},
  "Tag": "Required, but can be anything",
  "Children": [
    // a module,
   // a component,
   ....
   ...
  ]
}
```
*Components*

Component classes are looked up in this order:
1. Fully qualified name (crashes if it doesn't exist)
2. torchvision.models
3. torch.nn
4. json_torch_models.modules.default_modules

To add new components, add them to json_torch_models.modules.default_modules or modify json_torch_models.utils to map where you need.
```
{
  "ComponentClass": "SkippedLinker",
  "args": {
    "argument_a": "value_1"
    ....
    // this is unpacked as **kwargs to the module
  }
},
```

**UNet Syntax**

*Modules*
```
{
  "LogKwargs": {"log": "stuff", "not": "required"},
  "Tag": "Required, but can be anything",
  "Encoder": [List of modules and components],
  "Middle": [List of modules and components],
  "Decoder": [List of modules and components]
}
```
*Components*

No change

**Skipped Connections**

Skipped connection support is built in. In your component, add a "skipped_out" variable, or "skipped_in".
If you set "skipped_out": "a", then the variable "a" can be accessed (or overwitten!) by another component with "skipped_in".
If "skipped_in" is set, then two arguments will be passed to the ComponentClass's forward method. You should handle this with a custom class, or by using SkippedLinker
ComponentClass to wrap another module.

