from typing import Any, Optional

from toydl.core.scalar import Scalar


class Module:
    """
    Modules form a tree that stores parameters and other submodules.
    They make up the basis of neural network stacks.

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True  # TODO: 这里的 `training` 看起来并没有真正生效

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def train(self):
        """Set the mode of this module and all descendant modules to `train`."""
        self.training = True
        for m in self.modules():
            m.training = True

    def eval(self):
        """Set the mode of this module and all descendant modules to `eval`."""
        self.training = False
        for m in self.modules():
            m.training = False

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendants.


        :return list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        named_params = []
        # the module params
        for name, param in self._parameters.items():
            named_params.append((name, param))
        # descendants params
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                named_params.append((f"{module_name}.{param_name}", param))
        return named_params

    def parameters(self):
        """Enumerate over all the parameters of this module and its descendants."""
        params = [param for name, param in self.named_parameters()]
        return params

    def add_parameter(self, k: str, v: Scalar):
        """
        Manually add a parameter. Useful helper for scalar parameters.

        :param k: Local name of the parameter.
        :param v: Value for the parameter.
        :return parameter: Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        return super().__getattribute__(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, value: Scalar, name: Optional[str] = None):
        """
        :param value: the value of parameter
        :param name: the name of parameter
        """
        self.value = value
        self.name = name
        # 这里设置`requires_grad_`为 True 可以将当前参数的值，即对应的 Scalar instance
        self.value.requires_grad_(True)
        if self.name:
            self.value.name = self.name

    def update(self, x: Scalar) -> None:
        r"""Update the parameter value.

        :param x: the parameter's new value
        """
        self.value.data = x.data

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
