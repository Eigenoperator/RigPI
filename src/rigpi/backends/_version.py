"""Guard against Newton releases with incompatible spatial-vector conventions."""


def require_supported_newton(newton_module: object) -> None:
    version = getattr(newton_module, "__version__", None)
    if version != "1.6.1":
        raise RuntimeError(
            f"RigPI requires newton==1.6.1 for validated gradients; found {version!r}. "
            "Install the dependencies declared in pyproject.toml."
        )
