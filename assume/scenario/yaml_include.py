# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Minimal reimplementation of the YAML ``!include`` tag used to load AMIRIS
scenario files.

AMIRIS scenarios (see https://gitlab.com/dlr-ve/esy/amiris/examples) spread
their configuration across several YAML files and stitch them back together
with a custom ``!include`` tag, e.g.::

    Schema: !include "schema.yaml"
    Contracts: !include ["contracts/*.yaml", "Contracts"]

This used to be handled by the GPL-licensed ``pyyaml-include`` package,
which is incompatible with ASSUME's permissive distribution goals. This
module reimplements the (small) subset of that package's behaviour which
AMIRIS scenario files actually rely on:

* ``!include "relative/path.yaml"`` loads and parses a single file.
* Any ``!include`` urlpath containing a glob wildcard (``*``, ``?`` or
  ``[``) - whether written as a plain scalar or wrapped in a YAML sequence
  such as ``!include ["contracts/*.yaml", "Contracts"]`` - loads every file
  matching the pattern and returns their parsed contents as a list, sorted
  by path. Extra sequence entries beyond the urlpath (e.g. the ``"Contracts"``
  label above) are accepted for compatibility but carry no meaning here and
  are ignored. Matches are *not* flattened into a single list, even if every
  match is itself a YAML sequence.

All ``!include`` paths - including ones nested inside included files - are
resolved relative to the same fixed ``base_dir``, not relative to the file
they appear in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Characters that mark a urlpath as a glob pattern rather than a plain path.
WILDCARD_CHARS = ("*", "?", "[")


@dataclass
class Constructor:
    """PyYAML constructor implementing AMIRIS's ``!include`` tag.

    Register it with :func:`yaml.add_constructor` and use it to load a YAML
    file that contains ``!include`` tags::

        yaml.add_constructor("!include", Constructor(base_dir=base_path))
        with open(base_path + "/scenario.yaml", "rb") as f:
            scenario = yaml.load(f, Loader=yaml.FullLoader)
    """

    base_dir: str

    def __call__(self, loader: yaml.Loader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.ScalarNode):
            urlpath = loader.construct_scalar(node)
        elif isinstance(node, yaml.SequenceNode):
            # Additional entries are accepted for compatibility with AMIRIS
            # scenario files (e.g. `!include ["contracts/*.yaml", "Contracts"]`)
            # but carry no meaning here and are ignored.
            urlpath = loader.construct_sequence(node)[0]
        else:
            raise TypeError(f"!include does not support {type(node).__name__} nodes")

        base = Path(self.base_dir)
        loader_type = type(loader)
        
        if any(char in urlpath for char in WILDCARD_CHARS):
            matches = sorted(base.glob(urlpath))
            return [self._load(match, loader_type) for match in matches]
        return self._load(base / urlpath, loader_type)

    @staticmethod
    def _load(path: Path | str, loader_type: type) -> Any:
        with open(path, "rb") as f:
            return yaml.load(f, loader_type)
