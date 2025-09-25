def test_slice_extends():
    a = []

    def foo(*args):
        a.extend(args)

    foo(1, 2, 3)
    assert a == [1, 2, 3]
