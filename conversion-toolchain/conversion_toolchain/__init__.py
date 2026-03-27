import os

# Resolved at build time by setuptools (pyproject.toml: dynamic version = {attr = ...}).
# Falls back to "dev" when running from source without installing.
def _find_version():
    _here = os.path.dirname(__file__)
    # Check ../VERSION (Docker: /app/VERSION) then ../../VERSION (dev source tree)
    for rel in ('..', '../..'):
        candidate = os.path.join(_here, rel, 'VERSION')
        try:
            with open(candidate) as _f:
                return _f.read().strip()
        except FileNotFoundError:
            continue
    try:
        from importlib.metadata import version
        return version('oaax-nvidia-tensorrt-conversion')
    except Exception:
        return 'dev'

__version__ = _find_version()
