"""CLI wiring for search/report/verify: the arrival npz files and --zfw are
required (there is no flat-arrival fallback and TOW is an outcome of ZFW),
and the options reach run_search/run_report/run_verify unchanged.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from concopt.cli import main as cli_main

SEARCH_BASE = ["search", "--pln", "x.pln", "--decel", "D", "--npz", "x.npz",
               "--surface-npz", "x_surface.npz", "--zfw", "92.0"]
REPORT_BASE = ["report", "--pln", "x.pln", "--decel", "D", "--npz", "x.npz", "--surface-npz", "x_surface.npz",
               "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0"]
ARRIVAL_NPZS = ["--subsonic-npz", "x_subsonic.npz", "--arrival-upper-npz", "x_upper.npz"]


@pytest.mark.parametrize("base", [SEARCH_BASE, REPORT_BASE], ids=["search", "report"])
@pytest.mark.parametrize("given", [[], ARRIVAL_NPZS[:2], ARRIVAL_NPZS[2:]],
                          ids=["neither", "subsonic-only", "upper-only"])
def test_search_and_report_require_both_arrival_npzs(base, given):
    """The flat --decel-descent-min fallback is gone, so both arrival npz
    files are always needed."""
    with pytest.raises(SystemExit):
        cli_main([*base, *given])


@pytest.mark.parametrize("base", [SEARCH_BASE, REPORT_BASE], ids=["search", "report"])
def test_decel_descent_min_is_no_longer_an_option(base):
    with pytest.raises(SystemExit):
        cli_main([*base, *ARRIVAL_NPZS, "--decel-descent-min", "35"])


def test_search_report_pass_arrival_npzs_through():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main([*SEARCH_BASE, *ARRIVAL_NPZS])
    assert mock_search.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"
    assert mock_search.call_args.kwargs["arrival_upper_npz_path"] == "x_upper.npz"
    assert "decel_descent_min" not in mock_search.call_args.kwargs

    with patch("concopt.cli.run_report") as mock_report:
        cli_main([*REPORT_BASE, *ARRIVAL_NPZS])
    assert mock_report.call_args.kwargs["surface_npz_path"] == "x_surface.npz"
    assert mock_report.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"
    assert mock_report.call_args.kwargs["arrival_upper_npz_path"] == "x_upper.npz"
    assert "decel_descent_min" not in mock_report.call_args.kwargs


def test_zfw_is_required_and_tow_is_optional():
    """TOW is an outcome of ZFW: every weight-taking subcommand refuses to
    run without --zfw, while --tow alone is no substitute for it."""
    for argv in (["search", "--surface-npz", "s.npz", *ARRIVAL_NPZS],
                 ["report", "--date", "2020-01-01", "--hour", "10", "--surface-npz", "s.npz", *ARRIVAL_NPZS],
                 ["verify", "--date", "2020-01-01", "--hour", "10"]):
        with pytest.raises(SystemExit):
            cli_main([argv[0], "--pln", "x.pln", "--decel", "D", "--npz", "x.npz", *argv[1:], "--tow", "150"])


def test_verify_passes_zfw_and_subsonic_npz_through():
    with patch("concopt.cli.run_verify") as mock_verify:
        cli_main(["verify", "--pln", "x.pln", "--decel", "D", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10",
                  "--zfw", "92.0", "--subsonic-npz", "x_subsonic.npz"])

    assert mock_verify.call_args.kwargs["zfw_t"] == 92.0
    assert mock_verify.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"


def test_inflight_replay_speed_reaches_both_replay_sources_and_run_inflight(tmp_path):
    """C3 regression: --replay-speed has to reach BOTH replay_sources (the
    recording's own wall-clock -> flight-time mapping) AND run_inflight
    (its elapsed-time/sleep scaling) with the SAME value -- passing it to
    only one leaves the other at its default (1.0, real time), and the
    mismatch doesn't raise, it just makes the loop run far longer than the
    recording's own span and never reach touchdown (found live via this
    file's own CLI smoke test, where exactly that mismatch happened)."""
    recording_path = tmp_path / "recording.csv"
    pd.DataFrame({"elapsed_s": [0.0, 10.0]}).to_csv(recording_path, index=False)

    with patch("concopt.cli.replay_sources") as mock_replay_sources, \
         patch("concopt.cli.run_inflight") as mock_run_inflight:
        mock_replay_sources.return_value = ("state_source", "weather_source")
        cli_main(["inflight", "--pln", "x.pln", "--accel", "A", "--decel", "D", "--replay", str(recording_path),
                  "--replay-speed", "250"])

    assert mock_replay_sources.call_args.kwargs["replay_speed"] == 250.0
    assert mock_run_inflight.call_args.kwargs["replay_speed"] == 250.0
    assert mock_run_inflight.call_args.kwargs["state_source"] == "state_source"
    assert mock_run_inflight.call_args.kwargs["weather_source"] == "weather_source"


def test_inflight_without_replay_leaves_sources_none():
    """No --replay -> a real flight: state_source/weather_source must stay
    None so run_inflight connects to SimConnect/Active Sky for real (see
    run_inflight's own THE SEAM docstring)."""
    with patch("concopt.cli.run_inflight") as mock_run_inflight:
        cli_main(["inflight", "--pln", "x.pln", "--accel", "A", "--decel", "D"])

    assert mock_run_inflight.call_args.kwargs["state_source"] is None
    assert mock_run_inflight.call_args.kwargs["weather_source"] is None
    assert mock_run_inflight.call_args.kwargs["replay_speed"] == 1.0


def test_report_requires_surface_npz():
    """report adds the same runway penalties search does, so it needs the
    same surface-wind file."""
    import pytest
    with pytest.raises(SystemExit):
        cli_main(["report", "--pln", "x.pln", "--decel", "D", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0"])


@pytest.mark.parametrize("argv", [
    ["route", "--accel", "A"],
    ["search", "--npz", "x.npz", "--surface-npz", "s.npz", "--zfw", "92.0", *ARRIVAL_NPZS],
    ["report", "--npz", "x.npz", "--surface-npz", "s.npz", "--date", "2020-01-01", "--hour", "10",
     "--zfw", "92.0", *ARRIVAL_NPZS],
    ["verify", "--npz", "x.npz", "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0"],
    ["shortlist", "--npz", "x.npz"],
    ["inflight", "--accel", "A"],
], ids=["route", "search", "report", "verify", "shortlist", "inflight"])
def test_decel_has_no_default(argv):
    """The decel point (an `ATCWaypoint id`) is a property of the route, so
    every subcommand that takes --pln refuses to guess one."""
    with pytest.raises(SystemExit):
        cli_main([argv[0], "--pln", "x.pln", *argv[1:]])


@pytest.mark.parametrize("argv", [["route", "--decel", "D"], ["inflight", "--decel", "D"]])
def test_accel_has_no_default(argv):
    with pytest.raises(SystemExit):
        cli_main([argv[0], "--pln", "x.pln", *argv[1:]])


def test_decel_and_accel_reach_run_inflight_and_decel_reaches_run_search():
    with patch("concopt.cli.run_inflight") as mock_inflight:
        cli_main(["inflight", "--pln", "x.pln", "--accel", "A", "--decel", "D"])
    assert mock_inflight.call_args.args[1:3] == ("A", "D")

    with patch("concopt.cli.run_search") as mock_search:
        cli_main([*SEARCH_BASE, *ARRIVAL_NPZS])
    assert mock_search.call_args.kwargs["decel_id"] == "D"
