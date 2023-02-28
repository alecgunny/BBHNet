from dataclasses import dataclass

import numpy as np
import pytest

from bbhnet.analysis import events


class TestInjectionSet:
    def _test_read_write(self, obj, tmp_path):
        obj.write(tmp_path / "obj.h5")
        new = obj.__class__.read(tmp_path / "obj.h5")
        for key in obj.__dataclass_fields__:
            assert (getattr(obj, key) == getattr(new, key)).all()

    @pytest.fixture
    def parameter_set(self):
        @dataclass
        class Dummy(events.InjectionSet):
            ids: np.ndarray = events.parameter()
            age: np.ndarray = events.parameter()

        return Dummy

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        return tmp_path

    def test_parameter_set(self, parameter_set, tmp_dir):
        ids = np.array([1001, 1002, 1003])
        age = np.array([31, 35, 39])
        obj = parameter_set(ids, age)
        assert len(obj) == 3
        assert next(iter(obj)) == {"ids": 1001, "age": 31}

        subobj = obj[1]
        assert len(subobj) == 1
        assert subobj.ids[0] == 1002
        assert subobj.age[0] == 35

        subobj = obj[[0, 2]]
        assert len(subobj) == 2
        assert subobj.ids[-1] == 1003
        assert subobj.age[-1] == 39

        obj.append(obj)
        assert len(obj) == 6
        assert obj.ids[3] == 1001
        assert obj.age[3] == 31

        self._test_read_write(obj, tmp_dir)
