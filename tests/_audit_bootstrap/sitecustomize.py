"""Install the builder read recorder in child Python interpreters."""
import _artifact_audit

_artifact_audit.install()

# The CLI and other Python children use the same native-parser cache as pytest.
import os
if os.environ.get("HAENV_TEST_YAML_CACHE"):
    from pathlib import Path
    import _yaml_artifacts
    _yaml_artifacts._shared = Path(os.environ["HAENV_TEST_YAML_CACHE"])
    _yaml_artifacts.pytest_configure(None)
