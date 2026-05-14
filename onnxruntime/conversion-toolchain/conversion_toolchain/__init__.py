import os


def _find_version():
    _here = os.path.dirname(__file__)
    for rel in ('..', '../..', '../../..'):
        candidate = os.path.join(_here, rel, 'VERSION')
        try:
            with open(candidate) as _f:
                return _f.read().strip()
        except FileNotFoundError:
            continue
    try:
        from importlib.metadata import version
        return version('oaax-nvidia-ort-conversion')
    except Exception:
        return 'dev'


__version__ = _find_version()
