"""A configured Google budget never exceeds the provider's published output cap."""
from haenv.cli import load_cfg


def test_every_native_google_model_fits_its_official_output_cap():
    """A configured budget above the provider's maximum output is refused by native accounting."""
    from haenv.native_accounting import NativePrices
    models = load_cfg()["models"]
    seen = 0
    for key, spec in models.items():
        if spec.get("backend") != "google":
            continue
        cap = NativePrices.verified(spec["model"], "google", require_current=False).max_output_tokens
        assert spec["max_tokens"] <= cap, f"{key}: max_tokens {spec['max_tokens']} > official {cap}"
        seen += 1
    assert seen >= 3
