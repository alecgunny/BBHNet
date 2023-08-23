from inspect import Signature, signature


class Zoo:
    pass


class ZooMeta(type):
    def __new__(cls, name, bases, dct):
        arch = super().__new__(cls, name, bases, dct)

        parameters = signature(arch).parameters
        keys = list(parameters)
        new_params = [parameters[k] for k in keys[1:]]

        def arch_fn(*args, **kwargs):
            def f(num_ifos):
                return arch(num_ifos, *args, **kwargs)

            return f

        arch_fn.__signature__ = Signature(parameters=new_params)
        setattr(Zoo, arch.__name__, arch_fn)

        return arch
