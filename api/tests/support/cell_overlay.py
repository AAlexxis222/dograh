"""The cell overlay as written, for tests that pin its source (no Docker needed)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def overlay_source():
    """docker-compose.cell.yaml as written. Compose's own tags (!reset, !override) load as their plain value."""

    class Loader(yaml.SafeLoader):
        pass

    def plain(loader, _suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_scalar(node)

    Loader.add_multi_constructor("!", plain)
    return yaml.load(
        (ROOT / "docker-compose.cell.yaml").read_text(encoding="utf-8"), Loader=Loader
    )
