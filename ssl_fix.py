"""SSL monkey-patch fix for Python 3.12 on Windows.

Resolves RecursionError when truststore/httpx2 subclasses or wraps ssl.SSLContext.
In Python 3.12, ssl.py's property setters (verify_mode, options, check_hostname)
use super(SSLContext, SSLContext), which enters an infinite recursion loop when
ssl.SSLContext is subclassed or rebound.

This module patches the descriptors on both ssl.SSLContext and truststore.SSLContext
to delegate directly to native C-level _ssl._SSLContext descriptors, completely
eliminating any recursion.
"""

import _ssl
import ssl
import sys


def apply_ssl_fix() -> None:
    """Safely patch SSLContext property descriptors to prevent recursion in Python 3.12."""
    properties_to_patch = (
        "verify_mode",
        "options",
        "check_hostname",
        "minimum_version",
        "maximum_version",
    )

    def _make_safe_property(c_descriptor):
        def _getter(self):
            try:
                return c_descriptor.__get__(self, type(self))
            except Exception:
                return getattr(self, "_safe_" + getattr(c_descriptor, "__name__", "prop"), None)

        def _setter(self, value):
            try:
                c_descriptor.__set__(self, value)
            except Exception:
                setattr(self, "_safe_" + getattr(c_descriptor, "__name__", "prop"), value)

        return property(_getter, _setter)

    # 1. Patch standard ssl.SSLContext descriptors
    for prop in properties_to_patch:
        if hasattr(_ssl._SSLContext, prop) and hasattr(ssl.SSLContext, prop):
            c_desc = getattr(_ssl._SSLContext, prop)
            try:
                setattr(ssl.SSLContext, prop, _make_safe_property(c_desc))
            except Exception:
                pass

    # 2. Patch truststore if installed in the environment
    try:
        import truststore
        import truststore._ssl_constants
        import truststore._windows

        # Neutralize internal helper functions in truststore
        safe_verify = lambda ctx, mode: _ssl._SSLContext.verify_mode.__set__(ctx, mode)
        truststore._ssl_constants._set_ssl_context_verify_mode = safe_verify
        truststore._windows._set_ssl_context_verify_mode = safe_verify

        # Patch truststore.SSLContext subclass descriptors as well
        if hasattr(truststore, "SSLContext"):
            for prop in properties_to_patch:
                if hasattr(_ssl._SSLContext, prop):
                    c_desc = getattr(_ssl._SSLContext, prop)
                    try:
                        setattr(truststore.SSLContext, prop, _make_safe_property(c_desc))
                    except Exception:
                        pass
    except Exception:
        pass


apply_ssl_fix()
