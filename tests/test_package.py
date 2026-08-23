import dsfnet


def test_package_has_version():
    assert isinstance(dsfnet.__version__, str)
    assert dsfnet.__version__ == "0.1.0"
