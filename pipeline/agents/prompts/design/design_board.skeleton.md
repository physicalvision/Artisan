Use case: infographic-diagram
Asset type: engineering design board — the visual ground truth for 3D-modeling an articulated asset, read by a CAD/coding agent that also holds the exact numeric blueprint
Primary request: a clean, professional technical design board of {subject}
Style/medium: professional product / industrial-design technical design board — render the complete assembly and the orthographic views with REALISTIC materials and surface finish under soft, even studio lighting (gentle reflections, soft contact shadows); keep the exploded, kinematic and cutaway panels clean, evenly lit and technically legible (still in true material colour, readability first)
Composition/framing: one multi-panel reference sheet, panels clearly separated and each individually legible —
  1. orthographic FRONT, SIDE and TOP views of the complete, fully-assembled object;
  2. an exploded assembly view that separates every part along its assembly axis, each tagged with a thin leader line and a short label;
  3. a kinematic diagram of the moving parts in BOTH the closed AND the fully-open state, with each joint's axis and a motion-direction arrow;
  4. for any part that is hollow in reality, a SIMPLE schematic cutaway showing only its thin wall, its opening, and the blueprint part(s) in the cavity — no invented internal mechanism (sockets, wiring, brackets, etc.).
Use ONLY the panels above — do NOT add any extra panel (no separate parts-list, materials, or dimension/spec table); material and size info goes in labels/annotations, never its own panel.
Parts to depict: {parts_panel}
Articulation: {articulation}
Materials / finishes: {materials}
Color palette: {palette}
Scale: real-world scale in meters; annotate the overall bounding envelope ({key_proportions}) once — no per-part dimension table, those exact numbers live in the blueprint.
Constraints (the modeling truth — applies across every panel above):
  - ALL panels depict the SAME object: every panel shows the same parts as the orthographic views with identical shape/proportion/finish, only separated / posed / sectioned — never restyled or re-proportioned;
  - realistic materials and lighting are SKIN ONLY: they must not restyle, re-proportion, add, hide, or merge any part — use even, soft light and a neutral matte studio backdrop, never dramatic shadow that obscures geometry;
  - one-to-one with the blueprint parts: do NOT invent extra sub-parts, fasteners, screws, rivets, or decorative trim, and do NOT split or merge parts;
  - any part hollow in reality (shade, carcass, casing, cup, dome, housing) is a thin-wall shell with its real opening, never a solid block; whatever sits inside it is a separate part in the cavity;
  - every joint's pivot/axis lies on the real hinge/rail line, never at a part's center;
  - keep labels few, large, and legible, using exact blueprint names — rendered text is unreliable, so no dense small print.
Avoid: dramatic or dark cinematic lighting and heavy cast shadows that hide a part's true shape; invented decorative trim or fasteners not in the parts list; restyling or re-proportioning any part between panels; busy backgrounds, clutter, watermarks, brand logos, and any text other than part/section labels{avoid_extra}
