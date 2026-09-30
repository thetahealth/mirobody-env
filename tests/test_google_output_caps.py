"""Google output caps match the official model metadata; recorded budget observations keep their own basis."""
from haenv.cli import load_cfg
from haenv.llm import MEASURED_BUDGET_BASIS, recommended_budgets


def test_approved_google_caps_match_official_metadata_without_relabeling_history():
    models = load_cfg()["models"]
    for key in ("gemini-3.1-pro", "gemini-3.7-flash"):
        assert models[key]["backend"] == "google"
        assert models[key]["max_tokens"] == 65536
        assert MEASURED_BUDGET_BASIS[key]["cap"] == 72000
        assert models[key]["max_tokens"] >= recommended_budgets()[key]


def test_other_ranked_models_keep_their_approved_normal_budget():
    models = load_cfg()["models"]
    for key in ("deepseek-v4-pro", "deepseek-v4-flash", "gpt-6-luna", "gpt-6-sol",
                "minimax-m3", "glm-5.3-flash", "kimi-k3", "qwen3.7-flash"):
        assert models[key]["max_tokens"] == 72000


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


def test_gemini_flash_generation_cache_key_is_unchanged_by_the_cap():
    """The generation cache is keyed by the sampling conditions that differ from the model's
    default budget; the configured cap is the default for gemini-3.8-flash, so its sampling
    key stays empty."""
    from haenv.llm import keyed_sampling
    spec = load_cfg()["models"]["gemini-3.8-flash"]
    assert keyed_sampling(spec, "gemini-3.8-flash") == {}
