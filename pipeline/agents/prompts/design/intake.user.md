INTAKE subtask — UNDERSTAND the request.

This is the analysis stage: interpret what the user wants and how the real object works. Stay at the level of REQUIREMENTS and real-world understanding. Do NOT decompose into parts, assign dimensions, or define joints/axes/limits — that is the next stage (the blueprint).

Request prompt:
{prompt}

{reference_captions}

If reference images are attached, study each one (shape, proportions, seams, hinges, sliders, handles, likely hidden mechanisms) and apply the multi-image reading rules from your system prompt: decide per image whether it is a VIEWPOINT or an ARTICULATION/MOTION STATE (honor any text-prompt description; otherwise infer), treat the SAME part shown in different states as direct evidence of a moving part and its travel, and **never read the absence of visible motion as proof a part is fixed**. Combine the images with the prompt and real-world common sense.

Prepare a JSON object with these keys, then submit the complete object with the
`submit_intake(content)` tool:
- "name": short object name
- "category": object category (e.g. "cabinet", "table_lamp", "blender")
- "summary": one or two sentences — what this object is and what it is used for
- "key_interactions": list of the key movable / interactive behaviours the object needs, each in plain language combining the MOTION and its PURPOSE (e.g. "the lid opens upward to access the interior"). Describe motion + purpose only — do NOT name parts, joint types, axes, or limits (that is the blueprint's job). If one behaviour is physically conditional on another — a part can only move once a covering part is opened first (e.g. "the inner basket only slides out once the door is open", "the crisper drawer is reachable only with the fridge door open") — state that dependency in plain words; the blueprint turns it into an access-gate precondition.
- "scale_sense": one of "handheld" | "tabletop" | "furniture" | "room-scale" (a size category, NOT exact dimensions)
- "support_mode": one of "grounded" | "external_mount" | "mobile" — how the completed object is physically supported in normal use. Use "grounded" when it stands/rests on a floor or surface, "external_mount" when a wall/ceiling/host structure supports it, and "mobile" when rolling/translation is expected. Choose exactly one; do not infer a different mode downstream from the category name.
- "reference_observations": a STRUCTURED object describing what the images show (set to null if NO images are attached):
    - "multi_image_interpretation": one of "multiple_viewpoints" | "articulation_states" | "mixed" | "single" — your verdict on what the set is (prefer any description the text prompt gave; otherwise infer).
    - "per_reference": one entry per attached image, in order: {"ref": <1-based index>, "observed": "<form, proportions, materials, visible mechanisms for this image>", "depicted_state": "closed" | "open" | "partially_open" | "not_a_state" | "unknown"}.
    - "motion_evidence": ONLY when images show the SAME part in DIFFERENT states. One entry per observed moving part: {"moving_part": "<plain name>", "evidence": "<what changed between which refs>", "from_refs": [<indices>], "motion_kind": "rotation" | "translation", "candidate_axis_hint": "<plain words, e.g. 'horizontal hinge along the rear top edge'>", "candidate_range_hint": "<plain words, e.g. 'closed ~0° to open ~100°'>", "confidence": "high" | "medium" | "low"}. This is OBSERVATION, not engineering — do NOT name joint types, unit vectors, or radians. Leave it [] when you cannot see a part change state; absence here does NOT mean other parts are fixed (still capture their motion needs in "key_interactions").
- "assumptions": guesses you made about uncertain mechanism/form details, plus open questions to resolve (e.g. "assume a single rear hinge rather than a lift-off lid"). These are uncertain and may be revised downstream.

Rule of thumb: a behaviour you are confident the object needs → "key_interactions"; a detail you had to guess or that is ambiguous → "assumptions".

Do not stop with JSON in prose. Call `submit_intake`; if validation rejects it,
correct the complete object and resubmit.
