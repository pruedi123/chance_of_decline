import numpy as np
import pandas as pd
import streamlit as st

PRETTY_LBM = {
    "LBM 100E": "100% Equity",
    "LBM 90E": "90% Equity",
    "LBM 80E": "80% Equity",
    "LBM 70E": "70% Equity",
    "LBM 60E": "60% Equity",
    "LBM 50E": "50% Equity",
    "LBM 40E": "40% Equity",
    "LBM 30E": "30% Equity",
    "LBM 20E": "20% Equity",
    "LBM 10E": "10% Equity",
    "LBM 100F": "100% Fixed",
}
PRETTY_SPX = {f"spx{p}e": f"{p}% Equity" for p in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]}
PERIOD_STEP = 1  # monthly factors, advance one row per month

st.set_page_config(layout="wide")
st.title("Chance of Falling Below a Target")
st.caption("Estimate, for every allocation, the fraction of historical windows that finish below a selected target.")


@st.cache_data
def _load_factors(file_path: str, sheet: str, prefix: str) -> tuple[pd.DataFrame | None, list[dict]]:
    """Return (dataframe, allocation_meta). Each meta has raw + clean names. Cached to avoid re-reading files."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
    except Exception as exc:
        st.error(f"Unable to load {file_path} -> {sheet}: {exc}")
        return None, []
    df.columns = df.columns.astype(str).str.strip().str.replace("  ", " ")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    metas = []
    for col in df.columns:
        if col.upper().startswith(prefix):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            clean = col.strip()
            parts = clean.split()
            if len(parts) >= 2 and len(parts[-1]) == 1 and parts[-1].isalpha():
                clean = " ".join(parts[:-1]) + parts[-1]
            metas.append({"raw": col, "clean": clean})
    return df, metas


def _quantile_linear(arr: np.ndarray, q: float) -> float:
    """Helper to get a linear-interpolated quantile; works on NumPy ≥1.23 and older."""
    if arr.size == 0:
        return float("nan")
    try:
        return float(np.quantile(arr, q, method="linear"))
    except TypeError:
        return float(np.quantile(arr, q, interpolation="linear"))


def simulate_lump_values(factors: pd.Series, months: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-window lump-sum growth factors per $1 invested over 'months' months."""
    n = len(factors)
    months = int(months)
    if months <= 0 or n == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    # Fast path for monthly data (step == 1) using prefix sums of log factors.
    if step == 1:
        arr = factors.to_numpy(dtype=float, copy=False)
        valid = np.isfinite(arr) & (arr > 0)
        if len(arr) < months:
            return np.array([], dtype=float), np.array([], dtype=int)

        # Prefix sums to quickly get window log-products and invalid counts.
        log_vals = np.zeros_like(arr, dtype=float)
        log_vals[valid] = np.log(arr[valid])
        invalid_prefix = np.concatenate(([0], np.cumsum(~valid)))
        log_prefix = np.concatenate(([0.0], np.cumsum(log_vals)))

        window_end = np.arange(months - 1, n)
        window_start = window_end - (months - 1)

        invalid_in_window = invalid_prefix[window_end + 1] - invalid_prefix[window_start]
        ok_mask = invalid_in_window == 0
        if not np.any(ok_mask):
            return np.array([], dtype=float), np.array([], dtype=int)

        log_sum = log_prefix[window_end[ok_mask] + 1] - log_prefix[window_start[ok_mask]]
        values = np.exp(log_sum)
        return values.astype(float), window_start[ok_mask].astype(int)

    # Fallback for other step sizes (rare) uses the original loop.
    values = []
    starts = []
    max_start = n - (step * (months - 1))
    if max_start <= 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    for start in range(max_start):
        total = 1.0
        valid_window = True
        for period in range(months):
            idx = start + period * step
            if idx >= n:
                valid_window = False
                break
            f_val = factors.iloc[idx]
            if pd.isna(f_val) or f_val <= 0:
                valid_window = False
                break
            total *= float(f_val)
        if valid_window:
            values.append(total)
            starts.append(start)
    return np.array(values, dtype=float), np.array(starts, dtype=int)


def _pretty_name(source: str, clean: str) -> str:
    mapping = PRETTY_LBM if source == "Global" else PRETTY_SPX
    return mapping.get(clean, clean)


def _fmt_currency(val: float) -> str:
    if val is None:
        return "—"
    try:
        if not np.isfinite(val):
            return "—"
    except TypeError:
        return "—"
    return f"${val:,.0f}"


sb = st.sidebar
data_choice = sb.selectbox(
    "Data source",
    ["Global Equity", "S&P 500", "Both Global & SP500"],
    index=0,
    help="Choose which historical factor set(s) to evaluate.",
)
months_out = sb.number_input("Months from today", min_value=1, max_value=720, value=36)
current_value = sb.number_input(
    "Current portfolio value ($)", min_value=0, step=50_000, value=1_000_000, format="%i"
)
target_value = sb.number_input(
    "Target / floor value ($)",
    min_value=0,
    step=50_000,
    value=800_000,
    format="%i",
    help="Chance is computed on the current balance growing into the future and finishing below this amount.",
)
fee_pct = sb.slider(
    "Annual fee (%)",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.20,
    help="Net growth factor = gross factor × (1 − fee). Set to 0 to ignore fees.",
)

src_kind = (
    "BOTH"
    if data_choice.startswith("Both")
    else "LBM"
    if data_choice.startswith("Global")
    else "SPX"
)
df_lbm, metas_lbm = None, []
df_spx, metas_spx = None, []
if src_kind in ("LBM", "BOTH"):
    df_lbm, metas_lbm = _load_factors("global_mo_factors.xlsx", "factors_mo", "LBM")
if src_kind in ("SPX", "BOTH"):
    df_spx, metas_spx = _load_factors("spx_mo_factors.xlsx", "factors_mo", "SPX")

# Work on copies so cached data stays pristine across reruns with different fees.
if df_lbm is not None:
    df_lbm = df_lbm.copy()
if df_spx is not None:
    df_spx = df_spx.copy()

annual_fee = float(fee_pct) / 100.0
monthly_fee = annual_fee / 12.0
if annual_fee > 0:
    if df_lbm is not None and metas_lbm:
        df_lbm[[m["raw"] for m in metas_lbm]] = df_lbm[[m["raw"] for m in metas_lbm]] * (1.0 - monthly_fee)
    if df_spx is not None and metas_spx:
        df_spx[[m["raw"] for m in metas_spx]] = df_spx[[m["raw"] for m in metas_spx]] * (1.0 - monthly_fee)


def _selection_widget(label: str, metas: list[dict], mapping: dict[str, str]) -> list[dict]:
    if not metas:
        return []
    options = [m["clean"] for m in metas]
    selected = sb.multiselect(
        label,
        options=options,
        default=options,
        format_func=lambda clean: mapping.get(clean, clean),
    )
    if not selected:
        return []
    filtered = [m for m in metas if m["clean"] in selected]
    return filtered


if df_lbm is None and df_spx is None:
    st.stop()

selected_lbm = []
selected_spx = []
if df_lbm is not None:
    selected_lbm = _selection_widget("Global allocations", metas_lbm, PRETTY_LBM)
if df_spx is not None:
    selected_spx = _selection_widget("S&P 500 allocations", metas_spx, PRETTY_SPX)

if not current_value or current_value <= 0:
    st.warning("Enter a positive current portfolio value to see probabilities.")
if not selected_lbm and not selected_spx:
    st.warning("Pick at least one allocation in the sidebar.")


def _probability_rows(
    source_label: str,
    df_src: pd.DataFrame,
    metas: list[dict],
    months: int,
    curr: float,
    target: float,
) -> list[dict]:
    rows = []
    if df_src is None or not metas:
        return rows
    date_series = df_src["Date"] if "Date" in df_src.columns else None
    threshold_factor = float(target) / float(curr) if curr > 0 else float("nan")
    for meta in metas:
        col = meta["raw"]
        factors = df_src[col]
        lump_arr, start_idx = simulate_lump_values(factors, int(months), PERIOD_STEP)
        mask = np.isfinite(lump_arr)
        if not np.any(mask):
            continue
        lump_arr = lump_arr[mask]
        start_idx = start_idx[mask] if start_idx.size else np.array([], dtype=int)
        if np.isfinite(threshold_factor):
            prob_below = float(np.mean(lump_arr < threshold_factor))
            prob_above = float(np.mean(lump_arr >= threshold_factor))
        else:
            prob_below = float("nan")
            prob_above = float("nan")
        worst = float(np.min(lump_arr)) * curr if curr > 0 else float("nan")
        median = _quantile_linear(lump_arr, 0.5) * curr if curr > 0 else float("nan")
        p90 = _quantile_linear(lump_arr, 0.9) * curr if curr > 0 else float("nan")
        if start_idx.size == lump_arr.size:
            worst_start = int(start_idx[int(np.argmin(lump_arr))])
            if date_series is not None and 0 <= worst_start < len(date_series):
                date_val = date_series.iloc[worst_start]
                worst_label = date_val.strftime("%b %Y") if pd.notna(date_val) else "—"
            else:
                worst_label = f"Row {worst_start + 1}"
        else:
            worst_label = "—"
        rows.append(
            {
                "Source": source_label,
                "Allocation": _pretty_name(source_label, meta["clean"]),
                "Chance Below Target (%)": prob_below * 100.0,
                "Chance At/Above Target (%)": prob_above * 100.0,
                "Median Ending ($)": median,
                "90th % Ending ($)": p90,
                "Worst Ending ($)": worst,
                "Worst Window Start": worst_label,
                "# of Tests": int(lump_arr.size),
            }
        )
    return rows

result_rows: list[dict] = []
result_rows.extend(
    _probability_rows("Global", df_lbm, selected_lbm, months_out, current_value, target_value)
)
result_rows.extend(
    _probability_rows("SP500", df_spx, selected_spx, months_out, current_value, target_value)
)

if not result_rows:
    st.info("No valid simulations for the current selections.")
else:
    results_df = pd.DataFrame(result_rows)
    results_df.sort_values(by=["Source", "Chance Below Target (%)"], ascending=[True, False], inplace=True)
    if "Chance Below Target (%)" in results_df.columns:
        results_df["Chance At/Above Target (%)"] = 100.0 - results_df["Chance Below Target (%)"]
    display_df = results_df.copy()
    currency_cols = ["Median Ending ($)", "90th % Ending ($)", "Worst Ending ($)"]
    for col in currency_cols:
        if col in display_df:
            display_df[col] = display_df[col].apply(_fmt_currency)
    st.subheader("Historical chance of finishing below vs. at/above the target")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.Column(width="small"),
            "Allocation": st.column_config.Column(width="medium"),
            "Chance Below Target (%)": st.column_config.NumberColumn(format="%.1f%%", width="small"),
            "Chance At/Above Target (%)": st.column_config.NumberColumn(format="%.1f%%", width="small"),
            "Median Ending ($)": st.column_config.Column(width="medium"),
            "90th % Ending ($)": st.column_config.Column(width="medium"),
            "Worst Ending ($)": st.column_config.Column(width="medium"),
            "Worst Window Start": st.column_config.Column(width="small"),
            "# of Tests": st.column_config.Column(width="small"),
        },
    )

    explanation_lines: list[str] = []
    goal_text = f"Your goal is to have {_fmt_currency(target_value)} in {months_out} months."
    current_text = f"Your current portfolio is {_fmt_currency(current_value)}."
    explanation_lines.extend([goal_text, current_text])
    if target_value >= current_value:
        metric = "Chance At/Above Target (%)"
        series = results_df[metric].replace([np.inf, -np.inf], np.nan).dropna() if metric in results_df else pd.Series([], dtype=float)
        if not series.empty:
            best_idx = series.idxmax()
            best_row = results_df.loc[best_idx]
            best_prob = float(best_row[metric])
            allocation_desc = f"{best_row['Allocation']} ({best_row['Source']})"
            explanation_lines.append(
                f"The allocation that gives you the greatest chance of being at or above the goal is {allocation_desc}, "
                f"with roughly {best_prob:.1f}% of historical windows meeting or exceeding the target."
            )
    else:
        metric = "Chance Below Target (%)"
        series = results_df[metric].replace([np.inf, -np.inf], np.nan).dropna() if metric in results_df else pd.Series([], dtype=float)
        if not series.empty:
            best_idx = series.idxmin()
            best_row = results_df.loc[best_idx]
            prob_below = float(best_row[metric])
            prob_stay_above = max(0.0, 100.0 - prob_below)
            allocation_desc = f"{best_row['Allocation']} ({best_row['Source']})"
            explanation_lines.append(
                f"Because the target is below today’s balance, we focus on the chance of dipping under that floor. "
                f"{allocation_desc} keeps that risk to about {prob_below:.1f}% of the historical windows "
                f"({prob_stay_above:.1f}% stayed at or above the floor)."
            )
    if len(explanation_lines) >= 3:
        st.text("Quick takeaway: " + " ".join(explanation_lines))
