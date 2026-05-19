"""
Local shim for `pkg_resources` to use the vendored copy included with pip.

This is a minimal workaround for environments where `pkg_resources` is not
installed as a top-level package but is available under `pip._vendor.`

It exposes the vendored module's symbols at the top-level so `import
pkg_resources` succeeds for libraries that expect it.
"""
try:
    from pip._vendor import pkg_resources as _vendored_pkg_resources
except Exception:
    # Fall back to pip._internal.metadata pkg_resources if available
    try:
        from pip._internal.metadata import pkg_resources as _vendored_pkg_resources
    except Exception:
        raise ImportError("pkg_resources shim could not find a vendored pkg_resources")

# Re-export public attributes
for _name in dir(_vendored_pkg_resources):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_vendored_pkg_resources, _name)

__all__ = [n for n in dir() if not n.startswith("__")]
