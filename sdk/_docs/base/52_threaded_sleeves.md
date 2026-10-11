# Threaded Sleeves

Use `ThreadedSleeveFit` and `ThreadedSleeveGeometry` for a solid threaded
plug/socket fitting, such as an Edison-style light-bulb base and lamp socket,
that must retain a positive radial gap under full collision.

Do not use this helper for an open-mouth bottle neck/cap: the `plug` is solid and
the `socket` includes a closed top.

```python
from sdk import ThreadedSleeveFit, ThreadedSleeveGeometry, mesh_from_geometry

fit = ThreadedSleeveFit(
    male_radius=0.014,
    engagement_length=0.025,
    clearance=0.001,
)
bulb_base.visual(
    mesh_from_geometry(ThreadedSleeveGeometry(fit, role="plug"), "bulb_base.obj"),
    name="bulb_base_thread",
)
lamp_socket.visual(
    mesh_from_geometry(ThreadedSleeveGeometry(fit, role="socket"), "lamp_socket.obj"),
    name="lamp_socket_thread",
)
```

Both halves must use the same `fit`. The plug collider reaches the visual thread
crest, while the socket's segmented analytic wall begins exactly one `clearance`
beyond that radius. The socket therefore keeps its cavity instead of becoming a
filled convex hull.

The geometry is local-Z aligned. Put the joint on `axis=fit.axis`. Thread ridges
are visual; use the existing rotary-to-linear proportional coupling when rotation
must raise or lower the plug. The connector guarantees its own radial fit only;
the surrounding bulb envelope, socket housing, shoulder, and contact geometry
must still clear over the declared travel.
