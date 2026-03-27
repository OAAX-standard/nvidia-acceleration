import os

# Resolved at build time by setuptools (pyproject.toml: dynamic version = {attr = ...}).
# Falls back to "dev" when running from source without installing.
try:
    _version_file = os.path.join(os.path.dirname(__file__), '..', '..', 'VERSION')
    with open(_version_file) as _f:
        __version__ = _f.read().strip()
except FileNotFoundError:
    try:
        from importlib.metadata import version
        __version__ = version('oaax-nvidia-tensorrt-conversion')
    except Exception:
        __version__ = 'dev'
