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

from concopt.cli import main as cli_main


def test_search_report_decel_descent_defaults_agree():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main(["search", "--pln", "x.pln", "--npz", "x.npz",
                  "--surface-npz", "x_surface.npz"])
    search_decel_descent_min = mock_search.call_args.kwargs["decel_descent_min"]

    with patch("concopt.cli.run_report") as mock_report:
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10"])
    report_decel_descent_min = mock_report.call_args.kwargs["decel_descent_min"]

    assert search_decel_descent_min is None
    assert report_decel_descent_min is None


def test_search_report_pass_subsonic_npz_through():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main(["search", "--pln", "x.pln", "--npz", "x.npz",
                  "--surface-npz", "x_surface.npz",
                  "--subsonic-npz", "x_subsonic.npz"])
    assert mock_search.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"

    with patch("concopt.cli.run_report") as mock_report:
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10",
                  "--subsonic-npz", "x_subsonic.npz"])
    assert mock_report.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"


def test_verify_passes_zfw_and_subsonic_npz_through():
    with patch("concopt.cli.run_verify") as mock_verify:
        cli_main(["verify", "--pln", "x.pln", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10",
                  "--zfw", "92.0", "--subsonic-npz", "x_subsonic.npz"])

    assert mock_verify.call_args.kwargs["zfw_t"] == 92.0
    assert mock_verify.call_args.kwargs["subsonic_npz_path"] == "x_subsonic.npz"
