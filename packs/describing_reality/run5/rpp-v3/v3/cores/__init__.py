"""v3-native plain-language cores (structured prose and plain-text ontology).

These cores implement the v2 Arm interface like the archived v2 arms, but live in v3 because their behaviour is set by
genes (genes.core.params): `make_core(kind, params)` builds one per genome. v3.genome.render wraps them in a
RenderedArm exactly as it wraps a v2 arm looked up through the v2 registry (nothing is registered in rpp-core).

Both cores are built when the submitted document passes the core's checker with zero problems: validity() then
reports usable=True and executable=True (the "checked" document plays the role of an executable model), which is what
v3.episode.buildable() reads.
"""
from v3.cores.ontology import OntologyCore
from v3.cores.sectioned import SectionedCore
from v3.cores.structured import JsonCore, MarkdownCore

CORE_ARMS = {"sectioned": "SEC", "ontology": "ONT", "json": "JSN", "markdown": "MD"}   # kind -> render.CORE_SPECS
CLASSES = {"SEC": SectionedCore, "ONT": OntologyCore, "JSN": JsonCore, "MD": MarkdownCore}


def make_core(name, params):
    """Core name (CORE_ARMS value) + genes.core.params -> a core arm instance."""
    return CLASSES[name](params)
