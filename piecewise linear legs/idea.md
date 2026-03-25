Change-point segmentation on slope (piecewise linear legs)

NOTE - this will become a FAILURE. Documenting. The method can be useful for something else.

Idea
Price is modeled as piecewise linear with noise. Legs are segments with relatively stable slope. A zigzag turn is a change in slope sign and magnitude.

How to do it online
Maintain a rolling fit (or a Kalman local linear trend model) that estimates:
level_t, slope_t

A new leg begins when the posterior probability of slope sign flip is high AND the new slope magnitude is meaningfully different relative to recent noise.

No pips required. You work in units of uncertainty:
z_slope = slope_t / std(slope_t)

Turn condition example
sign(slope_t) != sign(slope_{t-1}) and abs(z_slope) >= z_min

This is a confidence threshold on a latent slope state.
