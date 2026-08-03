import pytest

from chiaroscuro_forge import optional as optional_module


def test_requires_optional_uses_is_available(monkeypatch):
    calls = []

    def fake_is_available(*deps):
        calls.append(deps)
        return False

    monkeypatch.setattr(optional_module, "is_available", fake_is_available)

    def fake_import_module(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(optional_module.importlib, "import_module", fake_import_module)

    @optional_module.requires_optional(
        "dummy", feature="REST API", install_hint="pip install dummy"
    )
    def decorated():
        return "ok"

    with pytest.raises(ImportError) as excinfo:
        decorated()

    message = str(excinfo.value)
    assert message == (
        "decorated requires missing packages: dummy. REST API is unavailable. "
        "Install with: pip install dummy"
    )
    assert calls == [("dummy",)]
