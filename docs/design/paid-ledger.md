# The paid-request ledger and request slots

Every paid request (a solver call or a judge call) does two things before it is sent: it
**reserves** its cost on a shared ledger, and it takes a **slot** (a lane in a pool). The
ledger bounds money, the slots bound concurrency. Both are shared by every process that uses
the same ledger path, so several runs can spend under one approved cap.

## Quick start

```bash
# A paid run names its ledger and cap: --judge-budget-ledger <dir>/budget.json --judge-budget-usd 40
haenv spend --ledger <dir>/budget.json                  # read-only dashboard, safe while a run is live
haenv ops --ledger <dir>/budget.json budget set 40      # raise (or lower) the cap, logged
haenv ops --ledger <dir>/budget.json capacity set 32    # change the shared slot ceiling now
haenv ops --ledger <dir>/budget.json drain              # stop dispatching; in-flight requests finish
```

Knobs live in `config.yaml` and have defaults, so none has to be set:

| Key | Role |
| --- | --- |
| `paid.global_request_limit`, `paid.judge_reserved_lanes` | Slots in the shared pool; lanes kept free of solver requests |
| `paid.backend_request_limits` | `backend: N` gives a backend its own pool |
| `paid.slot_poll_s` | How often the head of a waiting queue looks for a free lane |
| `paid.reserve_ahead` | Requests one process may hold a reservation for while they wait for a lane |
| `paid.tariff_expiry` | `google` / `dashscope` prices are checked against the vendors' pages on a dated day and valid for 7 days. `warn` (default) sends the request and warns once per backend; `refuse` stops it with "Tariff verification expired". **Official boards run with `refuse`** |
| `paid.tariff_checked_on` | `YYYY-MM-DD` on which you compared the built-in prices with the pricing pages; within 7 days of today there is no warning. Recorded as `tariff_check` in the run's price metadata |
| `budget.reservation_quantile`, `budget.reservation_min_observations`, `budget.overrun_slots` | Reservation sizing, below |
| `budget.pause_poll_s`, `budget.pause_timeout_s` | Waiting at the cap instead of failing |
| `budget.busy_timeout_s`, `budget.durable_commits` | Ledger storage |

The runtime document of a live run (`haenv ops ... runtime`) overrides capacity and the pause
knobs without a restart. `haenv.app.settings` declares each key with its default and role.

## The ledger

`budget.json` keeps its data in `budget.json.sqlite` (WAL): one row per request (status,
reserved and actual dollars, generation id, role, backend, model, timestamps) and a `meta`
table with the cap, the stop reason, the operator events and the running totals. Reserve,
settle and every other change is one `BEGIN IMMEDIATE` transaction that reads the totals,
checks the cap and writes the row and the new totals together. Writers queue on a kernel
lock, so a waiter wakes as soon as the lock is free and a killed process leaves the ledger as
it was. An older JSON ledger at the same path is imported once, read-only; `read_state`
reads either format.

`BudgetLedger.reserve` refuses at the cap (`BudgetCapReached`); `reserve_or_pause` waits for
room instead, and is the path solvers and judges both take.

## Reservations

A request reserves its worst case until its cost key (role, backend, model, request geometry)
has `reservation_min_observations` settled requests; after that it reserves their
`reservation_quantile`. Admission keeps `overrun_slots` times the largest (worst case minus
reservation) free under the cap, and a request that costs more than it reserved settles at its
true cost. Unknown costs are counted at the worst case. The limit is therefore never exceeded
by the requests the ledger admits, and the cap holds many more requests in flight than
worst-case reservations would.

## Order of one request

reserve (waits at the cap, holds no lane) → take a lane → write the `<receipt>.sent` marker →
send → release the lane → write the receipt → settle. A lane is held for the send only, so
ledger latency never occupies one, and requests queued for a lane already count against the
cap; a process lets at most `paid.reserve_ahead` requests hold a reservation without a lane, so
a start-up burst of hundreds of cells does not queue hundreds of ledger writes at once. A process killed while queued leaves a reservation with no marker: the resume sends it
under that reservation. A reservation with a marker and no answer is counted at its bound and
never sent again.

## Latency under load

A run has hundreds of threads in one process. A thread in the middle of a ledger transaction
has to retake the interpreter lock after every database call; with Python's default switch
interval (5 ms) it waits behind every runnable thread each time, and everything queued on the
transaction's write lock waits with it (measured: 12 processes with 6 busy threads each, p50
4 s per operation). `thread_switch_interval_s` (default 0.5 ms) sets the interval at the start
of a run; the same load then measures 54 ms. Progress logs and stop checks read the ledger's
running totals (`summary`), never every request (`snapshot`), whose cost grows with the ledger.

## Slots

A slot is a kernel lock on a lane file, released when its holder exits or dies. Callers that
may hold the same lanes wait in one queue (a kernel lock): only the head looks for a free lane,
every `paid.slot_poll_s`, and the rest sleep until their turn, so waiting costs no CPU and no
file reads. Raising a ceiling takes effect at once; lowering never preempts a request.
