Implement the blueprint below as `model.py`.

Attached images (in authority order; see the system prompt for how to weigh them):
{reference_captions}
- If USER REFERENCE photos are attached (labelled "Reference 1..N"), they are the AUTHORITATIVE truth for the object's real-world appearance and identity — match them. They may be different views and/or open/closed states of the SAME object; each is authoritative for the view/state it shows.
- If a `design_board.png` is attached, it is a GENERATED visual aid only (helpful, not authoritative).
- The BLUEPRINT below remains the binding SPEC (parts/dimensions/joints); where the board disagrees with it, the blueprint wins.

When the build compiles, also author `tasks.json` from the blueprint's `tasks` (see **Manipulation tasks** in your system prompt) before you stop.

BLUEPRINT (authoritative spec):
{blueprint}
