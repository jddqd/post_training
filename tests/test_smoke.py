import post_training


def test_package_imports() -> None:
    assert isinstance(post_training.__version__, str)
    assert post_training.__version__
