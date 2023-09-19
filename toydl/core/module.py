from typing import Any

from toydl.core.scalar import Scalar


class Module:
    """
    Modules form a tree that store parameters and other submodules.
    They make up the basis of neural network stacks.

    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def train(self):
        """Set the mode of this module and all descendent modules to `train`."""
        self.training = True
        for m in self.modules():
            m.training = True

    def eval(self):
        """Set the mode of this module and all descendent modules to `eval`."""
        self.training = False
        for m in self.modules():
            m.training = False

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.


        :return list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        named_params = []
        # the module params
        for name, param in self._parameters.items():
            named_params.append((name, param))
        # descendents params
        for module_name, module in self._modules.items():
            for param_name, param in module.named_parameters():
                named_params.append((f"{module_name}.{param_name}", param))
        return named_params

    def parameters(self):
        """Enumerate over all the parameters of this module and its descendents."""
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

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        assert False, "Not Implemented"

    def __repr__(self):
        def _add_indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None, name=None):
        """

        :param x: the value of parameter
        :param name: the name of parameter
        """
        self.value = x
        self.name = name
        # 这里设置`requires_grad_`为True可以将当前参数的值,即对应的Scalar instance
        self.value.requires_grad_(True)
        if self.name:
            self.value.name = self.name

    def update(self, x: Any):
        r"""Update the parameter value.

        ??? warning

            注意这里在`update`方法也调用了`self.value.requires_grad_(True)`方法来
            确保参数更新后依然是叶子节点，保证可以累计梯度计算并进行参数更新

        :param x: the parameter's new value
        """
        # self.value.data = x.data
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)
