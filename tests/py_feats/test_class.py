def test_class_default_variables():
    class Foo:
        foo: int = 123

        def __init__(self, val: int):
            self.foo = val

    f = Foo(234)
    assert f.foo == 234
    assert Foo.foo == 123
    f = Foo(345)
    assert f.foo == 345
    assert Foo.foo == 123


def test_dict_like_class_variables():
    class Foo:

        def __init__(self, data: dict):
            self.data = data

        def __getitem__(self, key: str):
            # or raise KeyError
            return self.data.get(key)

    foo = Foo({"foo": 123})
    assert foo['foo'] == 123
    assert foo['val'] is None
