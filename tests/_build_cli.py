"""Execute a test build with a stable revision label and the active interpreter."""
from functools import partial
import runpy
from haenv.app.artifacts import batch

batch.provenance_fields = partial(batch.provenance_fields, revision_label="component-test-build")
runpy.run_module("haenv", run_name="__main__")
