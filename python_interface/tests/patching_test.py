# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Tests for shared third-party patch registration helpers."""

# pylint: disable = missing-function-docstring

from types import ModuleType

import pytest

from aoclda._patching import register_patch, requested_patches


def test_register_patch_records_available_symbol():
    package = ModuleType("upstream")
    original = object()
    setattr(package, "available", original)
    replacement = object()
    registry = {}
    unavailable = {}

    register_patch(registry, unavailable, "available", package,
                   replacement)

    assert registry["available"] == {
        "pack": package,
        "orig_sym": original,
        "da_sym": replacement,
    }
    assert not unavailable


def test_unavailable_patch_warns_for_all_and_errors_when_explicit():
    package = ModuleType("upstream")
    registry = {}
    unavailable = {}

    register_patch(registry, unavailable, "missing", package,
                   object())

    assert not registry
    with pytest.warns(RuntimeWarning, match="upstream.missing is unavailable"):
        assert not list(requested_patches((), registry, unavailable))

    with pytest.raises(ValueError, match="upstream.missing is unavailable"):
        requested_patches(("missing",), registry, unavailable)


def test_unknown_explicit_patch_errors():
    with pytest.raises(ValueError, match="unknown"):
        requested_patches(("unknown",), {}, {})


def test_explicit_requests_are_validated_before_patching():
    registry = {"available": {}}

    with pytest.raises(ValueError, match="unknown"):
        requested_patches((["available", "unknown"],), registry, {})
