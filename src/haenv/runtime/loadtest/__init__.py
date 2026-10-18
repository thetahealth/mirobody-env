"""Offline rehearsal of a paid run: a fake provider and a driver that runs the real commands.

* `profile`  -- a measured provider profile (service times, error rates, price catalog) and
  the replay library of recorded answers, both built from finished batches;
* `responder` -- what the fake provider answers to one request;
* `server`   -- `FakeProvider`, a local HTTP server that speaks the OpenRouter, Google and
  DashScope APIs, sleeps the profiled service time and injects the profiled errors;
* `rehearsal` -- runs a rehearsal spec (jobs × models × k) against the fake provider, with
  an optional kill-and-resume, and writes the speed report.

The client side needs one setting, `transport.rehearsal_url`: every HTTPS request then goes
to the fake provider, so the whole client stack runs (pools, ledger, receipts, retries,
judges, reconcile) and nothing leaves the machine. See `docs/design/loadtest.md`.
"""
