import numpy as np
import pandas as pd


# ============================================================
# Kalman local linear trend zigzag on 5-minute bars
#
# What this does:
# 1. Builds a simple 2-state Kalman filter over price:
#       state = [level, slope]
# 2. Uses the filtered slope and its uncertainty to compute
#    a slope z-score at each bar.
# 3. Converts that z-score into a causal segmentation:
#       -1 = short leg
#        0 = balance / neutral
#       +1 = long leg
# 4. Returns:
#       df_leg: per-bar filter state + zigzag segmentation
#       legs:   one row per detected leg
#
# Important properties:
# - fully causal
# - no future lookahead
# - no pip reversal threshold
# - segmentation is driven by slope credibility, not price jumps
# ============================================================


def kalman_zigzag_5m(
    df5: pd.DataFrame,
    price_col: str = "mid_close",
    q_level: float = 1e-6,
    q_slope: float = 1e-8,
    r_meas: float = 1e-5,
    z_on: float = 2.0,
    z_flip: float = 3.0,
    min_leg_bars: int = 2,
    min_balance_bars: int = 1,
):
    """
    Segment a 5-minute price series into causal directional legs using a
    local linear trend Kalman filter.

    Parameters
    ----------
    df5 : pd.DataFrame
        Input 5-minute bar dataframe. Must contain:
        - timestamp
        - close_bid
        - close_ask
        If `price_col` is missing, mid_close is built from bid/ask close.

    price_col : str
        Column to use as the observed price series.

    q_level : float
        Process noise for the hidden level state.
        Higher values let the estimated level move more freely.

    q_slope : float
        Process noise for the hidden slope state.
        Higher values let the estimated trend change more quickly.

    r_meas : float
        Measurement noise for the observed price.

    z_on : float
        Absolute slope z-score needed to enter a directional leg.

    z_flip : float
        Stronger absolute slope z-score needed to flip directly from one
        directional leg to the opposite one.

    min_leg_bars : int
        Minimum number of bars a leg should last before a reversal is
        considered credible.

    min_balance_bars : int
        Minimum time spent in balance before a new leg can start.

    Returns
    -------
    df_leg : pd.DataFrame
        Per-bar output with:
        - timestamp
        - observed price
        - Kalman level
        - Kalman slope
        - slope standard deviation
        - slope z-score
        - zigzag state
        - leg id
        - leg start index
        - leg end index

    legs : pd.DataFrame
        Per-leg summary table with:
        - start/end index and timestamp
        - direction
        - duration
        - displacement
        - efficiency ratio
        - MAE / MFE
        - average slope diagnostics
    """

    # --------------------------------------------------------
    # 1. Prepare input price series
    # --------------------------------------------------------
    df = df5.copy()

    # If the requested price column does not exist, default to mid close.
    if price_col not in df.columns:
        df["mid_close"] = 0.5 * (
            df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)
        )
        price_col = "mid_close"

    y = df[price_col].to_numpy(float)
    n = len(y)

    if n < 5:
        raise RuntimeError("df5 too short for segmentation")

    # --------------------------------------------------------
    # 2. Define the Kalman model
    #
    # Hidden state:
    #   x_t = [level, slope]
    #
    # Transition:
    #   level_t = level_{t-1} + slope_{t-1} + noise
    #   slope_t = slope_{t-1} + noise
    #
    # Observation:
    #   observed_price_t = level_t + noise
    # --------------------------------------------------------
    F = np.array(
        [
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    H = np.array([[1.0, 0.0]], dtype=float)

    Q = np.array(
        [
            [q_level, 0.0],
            [0.0, q_slope],
        ],
        dtype=float,
    )

    R = np.array([[r_meas]], dtype=float)

    # --------------------------------------------------------
    # 3. Allocate storage for per-bar filter outputs
    # --------------------------------------------------------
    level = np.empty(n, dtype=float)
    slope = np.empty(n, dtype=float)
    slope_std = np.empty(n, dtype=float)
    z_slope = np.empty(n, dtype=float)

    # --------------------------------------------------------
    # 4. Initialize filter state
    #
    # Start at the first observed price with zero slope.
    # Covariance is initialized fairly simply here.
    # --------------------------------------------------------
    x = np.array([[y[0]], [0.0]], dtype=float)

    P = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    I = np.eye(2, dtype=float)

    # --------------------------------------------------------
    # 5. Run the Kalman filter forward in time
    #
    # This is fully causal:
    # each bar only uses information available up to that bar.
    # --------------------------------------------------------
    for t in range(n):
        if t > 0:
            # Predict next state from the previous state
            x = F @ x
            P = F @ P @ F.T + Q

        # Current observation
        y_t = np.array([[y[t]]], dtype=float)

        # Standard Kalman update
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        innovation = y_t - (H @ x)

        x = x + K @ innovation
        P = (I - K @ H) @ P

        # Save filtered outputs
        level[t] = float(x[0, 0])
        slope[t] = float(x[1, 0])

        # Guard against tiny negative numerical noise in covariance
        slope_var = max(P[1, 1], 1e-18)
        slope_std[t] = float(np.sqrt(slope_var))

        # "How credible is the slope relative to its uncertainty?"
        z_slope[t] = slope[t] / slope_std[t]

    # --------------------------------------------------------
    # 6. Convert slope z-score into a directional state
    #
    # State convention:
    #   -1 = short leg
    #    0 = balance / neutral
    #   +1 = long leg
    #
    # Hysteresis:
    # - z_on   starts a leg
    # - z_flip is stricter and allows an opposite flip
    #
    # This helps reduce noisy churning.
    # --------------------------------------------------------
    dir_state = np.zeros(n, dtype=np.int8)

    def _dir_from_z(z_value: float, threshold: float) -> int:
        """
        Convert a z-score into a direction using a symmetric threshold.
        """
        if z_value >= threshold:
            return 1
        if z_value <= -threshold:
            return -1
        return 0

    # Leg bookkeeping at the bar level
    leg_id = -np.ones(n, dtype=np.int64)
    leg_start = np.full(n, -1, dtype=np.int64)
    leg_end = np.full(n, -1, dtype=np.int64)

    # Current segmentation state
    cur_state = 0          # current direction: -1, 0, +1
    cur_leg = -1           # current leg id
    cur_start = -1         # bar index where current leg began
    last_switch_t = -10**9 # last bar where a new leg began

    # How long we have been sitting in balance
    balance_count = 0

    for t in range(n):
        if cur_state == 0:
            # ------------------------------------------------
            # We are currently in balance.
            # A new leg can only start if:
            # 1. |z_slope| is strong enough (>= z_on)
            # 2. we have stayed in balance long enough
            # ------------------------------------------------
            proposed_dir = _dir_from_z(z_slope[t], z_on)

            if proposed_dir == 0:
                dir_state[t] = 0
                balance_count += 1
                continue

            if balance_count < min_balance_bars:
                dir_state[t] = 0
                balance_count += 1
                continue

            # Start a new leg here
            cur_leg += 1
            cur_state = proposed_dir
            cur_start = t
            last_switch_t = t
            balance_count = 0

        else:
            # ------------------------------------------------
            # We are already inside a leg.
            #
            # Three possibilities:
            # 1. slope loses credibility -> go back to balance
            # 2. slope strongly supports opposite direction and
            #    dwell condition is met -> flip into new leg
            # 3. otherwise stay in current leg
            # ------------------------------------------------
            active_dir = _dir_from_z(z_slope[t], z_on)

            if active_dir == 0:
                # Slope is no longer strong enough to sustain the leg.
                # End the leg and move into balance.
                #
                # Note:
                # The original logic treats even too-short legs by simply
                # collapsing back to balance here rather than rewriting
                # earlier assignments.
                if (t - cur_start) >= min_leg_bars:
                    cur_state = 0
                    cur_start = -1
                    dir_state[t] = 0
                    balance_count = 1
                else:
                    cur_state = 0
                    cur_start = -1
                    dir_state[t] = 0
                    balance_count = 1
                continue

            # Check whether a full directional flip is allowed.
            flip_dir = _dir_from_z(z_slope[t], z_flip)

            can_flip = (
                flip_dir != 0
                and flip_dir != cur_state
                and (t - last_switch_t) >= min_leg_bars
            )

            if can_flip:
                # Close the old leg conceptually and start a new one here.
                cur_state = flip_dir
                cur_leg += 1
                cur_start = t
                last_switch_t = t
                dir_state[t] = cur_state
            else:
                # Continue current leg
                dir_state[t] = cur_state

        # Stamp the active leg fields on bars that belong to a leg
        if cur_state != 0:
            leg_id[t] = cur_leg
            leg_start[t] = cur_start

    # --------------------------------------------------------
    # 7. Backfill each bar's inclusive leg end index
    #
    # We scan contiguous runs of the same leg_id and stamp the
    # final bar index of that run into leg_end.
    # --------------------------------------------------------
    last_leg = -1
    last_t = -1

    for t in range(n):
        lid = int(leg_id[t])

        if lid == -1:
            continue

        if last_leg == -1:
            last_leg = lid
            last_t = t
        elif lid == last_leg:
            last_t = t
        else:
            leg_end[leg_id == last_leg] = last_t
            last_leg = lid
            last_t = t

    if last_leg != -1:
        leg_end[leg_id == last_leg] = last_t

    # --------------------------------------------------------
    # 8. Build the per-bar output dataframe
    # --------------------------------------------------------
    df_leg = df[["timestamp"]].copy()
    df_leg[price_col] = y
    df_leg["kf_level"] = level
    df_leg["kf_slope"] = slope
    df_leg["kf_slope_std"] = slope_std
    df_leg["kf_z_slope"] = z_slope
    df_leg["zz_state"] = dir_state
    df_leg["zz_leg_id"] = leg_id
    df_leg["zz_leg_start_i"] = leg_start
    df_leg["zz_leg_end_i"] = leg_end

    # --------------------------------------------------------
    # 9. Build one summary row per leg
    #
    # This is useful later for clustering, filtering, or
    # comparing leg structure.
    # --------------------------------------------------------
    leg_rows = []

    unique_leg_ids = sorted(int(x) for x in np.unique(leg_id) if x != -1)

    for lid in unique_leg_ids:
        idx = np.where(leg_id == lid)[0]
        if len(idx) == 0:
            continue

        s = int(idx[0])
        e = int(idx[-1])

        # Direction is inferred from the average filtered slope
        d = int(np.sign(np.nanmean(slope[idx])))

        p0 = float(y[s])
        p1 = float(y[e])
        disp = p1 - p0
        duration = e - s + 1

        # Efficiency ratio:
        # net movement divided by total path traveled
        path = float(np.sum(np.abs(np.diff(y[s:e + 1]))))
        net = float(abs(y[e] - y[s]))
        er = float(net / path) if path > 0 else 0.0

        # Segment prices
        seg = y[s:e + 1]

        # MAE / MFE are measured in raw price units here
        if d >= 0:
            # Long-style interpretation
            mae = float(max(0.0, p0 - np.min(seg)))
            mfe = float(max(0.0, np.max(seg) - p0))
        else:
            # Short-style interpretation
            mae = float(max(0.0, np.max(seg) - p0))
            mfe = float(max(0.0, p0 - np.min(seg)))

        leg_rows.append(
            {
                "leg_id": lid,
                "dir": d,
                "start_i": s,
                "end_i": e,
                "start_ts": df_leg["timestamp"].iloc[s],
                "end_ts": df_leg["timestamp"].iloc[e],
                "duration_bars": duration,
                "disp": disp,
                "abs_disp": abs(disp),
                "er": er,
                "mae": mae,
                "mfe": mfe,
                "slope_mean": float(np.mean(slope[idx])),
                "slope_std_mean": float(np.mean(slope_std[idx])),
                "z_slope_mean": float(np.mean(np.abs(z_slope[idx]))),
            }
        )

    legs = pd.DataFrame(leg_rows)

    return df_leg, legs


# ============================================================
# Example run on df5
# ============================================================
df_leg, legs = kalman_zigzag_5m(
    df5,
    price_col="mid_close",   # if missing, it will be created from bid/ask close
    q_level=1e-6,
    q_slope=1e-8,
    r_meas=1e-5,
    z_on=2.0,
    z_flip=3.0,
    min_leg_bars=2,
    min_balance_bars=1,
)

print("[KF_ZZ] bars:", len(df_leg), "| legs:", len(legs))
print("[KF_ZZ] zz_state counts:")
print(df_leg["zz_state"].value_counts(dropna=False).sort_index())
print("[KF_ZZ] legs head:")
print(legs.head(10))