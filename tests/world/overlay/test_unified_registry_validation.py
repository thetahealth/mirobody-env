"""All authored conditions must participate in registry and scoring checks."""
import pytest
import haenv.world.overlay as O
from haenv.scoring.matching.aliases import alias_hit_asserted

def test_unified_alias_collisions_are_validated(monkeypatch):
    kernel = O._kernel_specs()
    spec = dict(next(iter(O.HAENV_UNIFIED_SPECS.values())))
    spec["aliases"] = list(kernel["JD-HYPO"]["aliases"])
    monkeypatch.setattr(O, "HAENV_UNIFIED_SPECS", {"HD-UNI-99": spec})
    O.validate_registry.cache_clear()
    try:
        with pytest.raises(O.RegistryInvalid, match="check_alias_collisions"):
            O.validate_registry()
    finally:
        O.validate_registry.cache_clear()

@pytest.mark.parametrize("name", ["胰腺癌相关糖尿病", "3c型糖尿病", "pancreatic cancer associated diabetes"])
def test_pancreatic_diabetes_does_not_hit_generic_ir_alias(name):
    aliases = O.condition_registry()["JD-IR"]["aliases"]
    assert not alias_hit_asserted(name, aliases)

def test_separate_generic_diabetes_mention_still_counts():
    assert alias_hit_asserted("胰腺癌相关糖尿病之外另列2型糖尿病", ["糖尿病"])

def test_all_authored_unified_conditions_validate():
    assert O.validate_registry()
