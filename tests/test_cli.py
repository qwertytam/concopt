"""concopt search and concopt report both describe the decel-point-to-
touchdown segment. B3 replaced the old flat DECEL_DESCENT_S constant with
arrival.arrival() as the default for both; --decel-descent-min is now an
explicit opt-in override (forcing the flat legacy arrival) rather than a
value with its own default, so this guards that both subcommands agree it
is None (real model) unless the user asks otherwise -- the same "the two
CLIs must agree" property the old test checked, updated for the new
contract.
"""
from unittest.mock import patch

import pandas as pd

from concopt.cli import main as cli_main


def test_search_report_decel_descent_defaults_agree():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main(["search", "--pln", "x.pln", "--npz", "x.npz",
                  "--surface-npz", "x_surface.npz", "--zfw", "92.0"])
    search_decel_descent_min = mock_search.call_args.kwargs["decel_descent_min"]

    with patch("concopt.cli.run_report") as mock_report:
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz", "--surface-npz", "x_surface.npz",
                  "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0"])
    report_decel_descent_min = mock_report.call_args.kwargs["decel_descent_min"]

    assert search_decel_descent_min is None
    assert report_decel_descent_min is None


def test_search_report_pass_subsonic_npz_through():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main(["search", "--pln", "x.pln", "--npz", "x.npz",
                  "--surface-npz", "x_surface.npz", "--zfw", "92.0",
                  "--subsonic-npz", "x_subsonic.npz"])
    assert mock_search.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"

    with patch("concopt.cli.run_report") as mock_report:
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz", "--surface-npz", "x_surface.npz",
                  "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0",
                  "--subsonic-npz", "x_subsonic.npz"])
    assert mock_report.call_args.kwargs["surface_npz_path"] == "x_surface.npz"
    assert mock_report.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"


def test_zfw_is_required_and_tow_is_optional():
    """TOW is an outcome of ZFW: every weight-taking subcommand refuses to
    run without --zfw, while --tow alone is no substitute for it."""
    import pytest
    for argv in (["search", "--surface-npz", "s.npz"],
                 ["report", "--date", "2020-01-01", "--hour", "10"],
                 ["verify", "--date", "2020-01-01", "--hour", "10"]):
        with pytest.raises(SystemExit):
            cli_main([argv[0], "--pln", "x.pln", "--npz", "x.npz", *argv[1:], "--tow", "150"])


def test_verify_passes_zfw_and_subsonic_npz_through():
    with patch("concopt.cli.run_verify") as mock_verify:
        cli_main(["verify", "--pln", "x.pln", "--npz", "x.npz",
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
        cli_main(["inflight", "--pln", "x.pln", "--replay", str(recording_path),
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
        cli_main(["inflight", "--pln", "x.pln"])

    assert mock_run_inflight.call_args.kwargs["state_source"] is None
    assert mock_run_inflight.call_args.kwargs["weather_source"] is None
    assert mock_run_inflight.call_args.kwargs["replay_speed"] == 1.0


def test_report_requires_surface_npz():
    """report adds the same runway penalties search does, so it needs the
    same surface-wind file."""
    import pytest
    with pytest.raises(SystemExit):
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10", "--zfw", "92.0"])
