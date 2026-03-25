1. Define the leg state (what exists at every bar)

At every finalized 5m bar t maintain:
level estimate L_t
slope estimate S_t
slope uncertainty Var(S_t) or Std(S_t)
current leg direction dir_t = sign(S_t) when |S_t| is credible, otherwise 0
current leg start index s
current leg running features (duration, displacement, efficiency, MAE, etc.)

We get L_t, S_t, Var(S_t) from a local linear trend Kalman filter, or from a rolling regression plus an online variance estimator. Kalman is cleaner because it gives uncertainty for free.

2. Define what counts as “credible slope” without pip thresholds

Avoid pips by working in z space:

z_t = S_t / Std(S_t)

Interpretation:

z_t near 0 means direction is not distinguishable from noise
|z_t| large means the drift is real relative to the current noise level

Structure rule:

dir_t = sign(S_t) only when |z_t| >= z_on
otherwise dir_t = 0 (balance, no leg)

This is still a threshold, but it is not a pip threshold and it adapts to volatility because Std(S_t) changes.

3. Define leg start and leg end causally

Leg start at time t when:

dir_{t-1} is 0 (or opposite), and dir_t becomes +1 or -1
plus a persistence check so we do not flip on a single bar (for example require dir_t stays same for k bars, or require z_t stays beyond z_on for k bars)

Leg end at time t when:

dir_t becomes 0 (trend loses credibility), or
dir_t flips sign with credibility (|z_t| >= z_flip) and persists

This yields a zigzag as a sequence of legs, each leg defined by a start time and an end time in forward time.

4. Add the minimum amount of “anti flip” logic

This is where huge chance to create garbage. Keeping it minimal and explicit:

hysteresis: z_flip > z_on
meaning it is easier to start a leg than to reverse it, or the opposite, but choose one and freeze it

dwell: once a leg starts, require at least m bars before allowing a flip, unless z is extremely strong

These are structural stabilizers, not pip gates.

What we can measure per leg with zero labels

Once legs exist, we compute leg descriptors, all causal:

duration_bars
signed displacement from leg start to now
efficiency ratio (net/path) over the leg so far
MAE so far relative to leg direction
volatility ratio inside leg versus pre leg
slope stability (variance of S within the leg)

These are the raw materials for clustering later, and they are available in forward replay without any labels.


MORE NOTES (AI generated summary)

1. What the model is actually doing

In a local linear trend (Kalman) view, price is decomposed into:

level_t

slope_t

A zigzag leg is simply a contiguous interval where:

slope_t has consistent sign

slope magnitude is materially different from noise

A turn occurs when:

slope sign flips

and the new slope magnitude is statistically credible

So the zigzag is not based on price extremes.
It is based on regime of directional drift.

2. What naturally clusters

Once you segment into legs, you will find legs cluster in a feature space.

Each leg can be represented by:

mean slope magnitude

duration (bars)

cumulative displacement

volatility during leg

efficiency ratio

slope stability (variance of slope_t)

When you cluster those, you usually see:

Cluster A: Strong directional impulse

high |slope|

long duration

high efficiency

low internal variance

Cluster B: Short burst / spike

high |slope|

short duration

medium efficiency

often volatility expansion

Cluster C: Drift

low |slope|

long duration

moderate efficiency

low volatility

Cluster D: Chop / false leg

small |slope|

short duration

low efficiency

high internal reversals

These clusters are not invented. They emerge naturally when you embed legs in slope-duration-volatility space.

3. How clusters relate sequentially

Now this is where it becomes interesting.

Leg clusters do not occur randomly.

Empirically, you often observe:

Strong impulse → short counter impulse → continuation impulse

Strong impulse → volatility compression → regime shift

Drift → impulse expansion

Chop → chop → breakout impulse

This means zigzag is not just alternating direction.

It is a Markov-like process over leg types.

You can model transitions:

P(cluster_j | cluster_i)

That gives structural behavior, not just geometry.

4. Why this is better than pip zigzag

A pip-based zigzag forces symmetry.

A slope-based segmentation allows:

long shallow trends

fast violent spikes

compressed sideways drifts

All under the same framework.

And clustering lets you answer:

When is a leg statistically similar to past profitable legs?

That is where “get on / get off” becomes grounded.

5. How the clustering really works geometrically

Imagine 3 axes:

X = normalized slope magnitude

Y = duration

Z = efficiency ratio

Legs become points in 3D.

They naturally form dense regions.

You are not clustering price.
You are clustering behaviors.

And those behaviors repeat.

6. The key conceptual shift

Zigzag is not:

“price reversed”

It is:

“drift regime changed”

Clustering reveals:

which drift regimes matter

which are noise

which precede continuation

which precede collapse