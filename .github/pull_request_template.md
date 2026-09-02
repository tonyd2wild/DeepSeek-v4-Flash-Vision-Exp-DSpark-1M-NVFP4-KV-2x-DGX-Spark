<!-- This repo keeps ONE current recipe, written down in CURRENT.md. A PR is checked against that file, not against the whole tree. -->

**Which line of CURRENT.md does this change?**
(quote it, or write "none: docs/tooling only")

**What is the new value, and what number proves it?**
(flag, before -> after, the prompt type and temperature of the measurement, and where the log or JSON lives)

**Tested on**
(hardware, image tag, weights dir, date)

**Checklist**
- [ ] I rebased on current `main` (stale PRs merge old configs over the current recipe)
- [ ] `CURRENT.md` updated in this PR if a launcher or a serving flag changed, and `bash tools/check-current.sh --write` was run
- [ ] Speed numbers are from real prompts (prose, code, ...); any counting-prompt number is labeled as the draft-acceptance ceiling; prefill is cold
