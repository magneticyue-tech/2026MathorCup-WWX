# Review module

Cross-team communication for this repo. One markdown file per thread, named
`YYYY-MM-DD-short-slug.md`, created from `review/_TEMPLATE.md`.

## How it works
- File a thread when one team needs another to act or decide (e.g. Coding
  reports a solver anomaly to Modeling).
- Set `from`, `to`, `severity`, and `refs` in the frontmatter so the right team
  can find it and open the referenced artifacts.
- Lifecycle: `open` -> `ack` (recipient has seen it) -> `resolved`.
- All agents scan `review/` for open threads addressed to their team at the
  start of a session (see `COLLABORATION.md`).
- Always communicate with user before filing anything. 

## Promotion to memory
When a resolved thread carries a lesson that should outlive it (a new
constraint, a convention), create a `memory/` file for it and link that file
from the thread's Resolution section. Threads are conversation; `memory/` is
the durable record.
