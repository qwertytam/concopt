"""Acceptance tests for search.run_shortlist -- the pure CSV-in,
printed-commands-out helper behind `concopt shortlist`.
"""
import pandas as pd

from concopt.search import DEFAULT_TOW_T, run_shortlist


def _sample_results_csv(path, n=15):
    """A minimal concopt search --out CSV -- just the columns run_shortlist
    reads (date, local_departure, tow_t, total_time, mean_fl, mean_wind_kt,
    flags), already sorted best-first like the real output."""
    df = pd.DataFrame({
        "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "local_departure": ["14:00"] * n,
        "total_time": ["3:0" + str(i % 10) for i in range(n)],
        "mean_fl": [507] * n,
        "mean_wind_kt": [76.3] * n,
        "flags": [""] * n,
        "tow_t": [185.0] * n,
    })
    df.to_csv(path, index=False)
    return df


def test_run_shortlist_returns_top_n_rows(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=15)

    out = run_shortlist(csv_path, "route.pln", "route_legs.npz", top=5)

    assert len(out) == 5
    assert out["date"].tolist() == [f"2026-01-{i + 1:02d}" for i in range(5)]


def test_run_shortlist_prints_ready_to_run_verify_command(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=3)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=3, decel_id="BARIX")

    printed = capsys.readouterr().out
    assert "concopt verify --pln route.pln --npz route_legs.npz" in printed
    assert "--date 2026-01-01 --hour 14" in printed
    assert f"--tow {DEFAULT_TOW_T:.0f}" in printed
    assert "--decel BARIX" in printed


def test_run_shortlist_uses_each_row_own_tow(tmp_path, capsys):
    """A shortlist spanning search runs made under different --tow values
    should generate a command matching each row's own tow_t, not a shared
    default."""
    csv_path = tmp_path / "results.csv"
    df = _sample_results_csv(csv_path, n=2)
    df.loc[1, "tow_t"] = 175.0
    df.to_csv(csv_path, index=False)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=2)
    printed = capsys.readouterr().out

    assert "--tow 185" in printed
    assert "--tow 175" in printed


def test_run_shortlist_blank_flags_shown_as_dash(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _sample_results_csv(csv_path, n=1)

    run_shortlist(csv_path, "route.pln", "route_legs.npz", top=1)

    assert "flags: -" in capsys.readouterr().out
