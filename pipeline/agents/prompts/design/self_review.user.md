BLUEPRINT SELF-REVIEW — round {review_round}.

Review the exact saved Blueprint below before it is released. This is primarily
your judgement, not a deterministic checklist exercise. Re-read the original P0
intake and P1 authoring conversation that precede this message, then decide
whether the Blueprint faithfully and coherently captures the requested object.

Pay particular attention to:
- fidelity to the request, asset brief, and any attached reference evidence;
- whether all necessary movable behavior and physical preconditions are covered,
  without adding speculative or unnecessary custom mechanisms;
- access gates in BOTH directions: whether a movable part that is physically
  blocked until another part moves aside is missing its `motion_gate`, and
  whether a declared gate has no real occlusion behind it. Both errors are real.
  Most objects legitimately have none, so absence is a finding only when the
  object's own physics says a part is blocked;
- consistency among parts, hierarchy, joints, dimensions, materials, tasks, and
  acceptance criteria;
- whether acceptance criteria describe observable physical outcomes that Coding
  and Review can act on, rather than prescribing low-level implementation.

Do not inspect the SDK in this stage and do not add fields to the Blueprint for
the sake of this review. Use your engineering judgement. Minor stylistic
preferences are not REDO reasons.

Current Blueprint SHA-256:
{blueprint_sha256}

Current saved Blueprint:
{blueprint}

Submit exactly one verdict with `submit_blueprint_selfcheck`:
- `PASS`: every check verdict is PASS and `findings` is empty. This releases P1.
- `REDO`: identify concrete unresolved findings. After the tool response, revise
  the Blueprint in this same conversation and submit the complete replacement
  with `write_blueprint`; it will be reviewed again.

Use concise `checks` entries of `{ "area", "verdict", "rationale" }`. Each
finding must contain non-empty `problem`, `evidence`, and `required_revision`
strings. Set `blueprint_sha256` to the exact hash above.
