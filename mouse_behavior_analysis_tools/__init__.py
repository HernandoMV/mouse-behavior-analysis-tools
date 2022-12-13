from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mouse-behavior-analysis-tools")
except PackageNotFoundError:
    # package is not installed
    pass
