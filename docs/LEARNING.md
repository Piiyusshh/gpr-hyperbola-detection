# Step 1 learning notes

## GPR in plain language

Ground Penetrating Radar sends a short electromagnetic pulse into the ground
and records echoes from places where electrical properties change. A metal
pipe reflects strongly because it is very different electrically from soil.

An **A-scan** is one recorded trace: signal strength against time at one
antenna position. A **B-scan** is many neighbouring A-scans arranged side by
side, so it shows time (and approximately depth) against position.

A point-like target such as the top of a pipe is closest when the antenna is
directly over it. At positions to either side, the wave must travel farther
along a sloping path, so the echo arrives later. Plotting those arrival times
forms a hyperbola. The apex is the earliest arrival, at the target's lateral
position.

Relative permittivity tells us how much slower an electromagnetic wave travels
in a material than in vacuum. For a simple non-magnetic material,
`v = c/sqrt(epsilon_r)`. A larger relative permittivity means a slower wave,
so the same echo time corresponds to a shallower depth. Getting permittivity
wrong is therefore a direct depth error.
