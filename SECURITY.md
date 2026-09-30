# Security Policy

## Supported versions

HAEnv is pre-1.0. Only the latest release on `main` receives fixes.

| Version | Supported |
|---|---|
| latest `main` | ✅ |
| anything older | ❌ |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: the *Security* tab of this
repository, "Report a vulnerability". It creates a private thread with the
maintainers; you do not need an email address and nothing is public while it is open.

If you cannot use that form, open a normal issue that says only *"I need a private
channel for a security report"* (do not put the details in it) and a maintainer
will open the private thread for you.

Please include, as far as you can: what you did, what happened, what you expected,
the commit or release you were on, and your Python version. A proof of concept helps
but is not required.

We will acknowledge within a few working days and tell you whether we consider it in
scope. There is no bounty program.

## Also report these here

HAEnv ships synthetic clinical data and a scoring kernel. The failure modes below are
not classic vulnerabilities, and we want to hear about them privately rather than in a
public issue:

| If you believe… | Report it privately, the same way |
|---|---|
| any shipped case is not synthetic: that it resembles a real patient, or that dates/identifiers look like they came from a real record | yes |
| any shipped text is a third party's: a verbatim case report, a figure, a vocabulary, or anything else we had no right to redistribute under [`LICENSE`](LICENSE) / [`LICENSE-DATA`](LICENSE-DATA) | yes |
| a credential, key, internal hostname or absolute path is present in the repository, its history, or a published artifact | yes |
| a published score or leaderboard can be forged or replayed past the publish gate | yes |

We treat all four as security-class reports: private thread first, fix, then a public
entry in [`CHANGELOG.md`](CHANGELOG.md) once the fix has shipped.

## Out of scope

* Anything about the clinical correctness of a case. This is evaluation data and is
  not medical advice; see [`docs/ETHICS.md`](docs/ETHICS.md). Clinical objections
  belong in a normal issue and are welcome there.
* Denial of service against your own machine by feeding the generator absurd parameters.
* Findings that require you to already control the machine running HAEnv.
* Vulnerabilities in a model provider's API. Report those to the provider.

## What HAEnv does with your secrets

The harness reads model credentials from an environment file whose path is configured
in `config.yaml`; it never writes them to logs or to any artifact under `results/` or
`reports/`. If you find a credential in an artifact, that is the first row of the table
above — please report it.
