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

"""Shared helpers for registering and selecting third-party patches."""

import warnings


def register_patch(registry, unavailable, name, package, da_symbol):
    """Register a patch if its original symbol exists in the upstream package."""
    try:
        original_symbol = getattr(package, name)
    except AttributeError:
        unavailable[name] = f"{package.__name__}.{name} is unavailable"
        return

    registry[name] = {
        'pack': package,
        'orig_sym': original_symbol,
        'da_sym': da_symbol,
    }


def requested_patches(args, registry, unavailable):
    """Resolve requested patch names, warning for patch-all and rejecting bad names."""
    if not args:
        for reason in unavailable.values():
            warnings.warn(reason, RuntimeWarning, stacklevel=3)
        return registry.keys()

    requested = args[0]
    if isinstance(requested, str):
        names = [requested]
    elif isinstance(requested, (list, tuple)):
        names = requested
    else:
        raise TypeError("Unrecognized argument")

    for name in names:
        if name not in registry:
            reason = unavailable.get(
                name, f"No AOCL-DA patch is registered for {name!r}")
            raise ValueError(reason)

    return names
