"""concopt search and concopt report both describe the decel-point-to-
touchdown segment (search.DECEL_DESCENT_S); report.py used to carry its own,
disagreeing default (report.DEFAULT_DECEL_DESCENT_S, 17.1 min vs 35 min) --
this guards against that drifting apart again."""
from unittest.mock import patch

from concopt.cli import main as cli_main
from concopt.search import DECEL_DESCENT_S


def test_search_report_decel_descent_defaults_agree():
    with patch("concopt.cli.run_search") as mock_search:
        cli_main(["search", "--pln", "x.pln", "--npz", "x.npz",
                  "--surface-npz", "x_surface.npz"])
    search_decel_descent_s = mock_search.call_args.kwargs["decel_descent_s"]

    with patch("concopt.cli.run_report") as mock_report:
        cli_main(["report", "--pln", "x.pln", "--npz", "x.npz",
                  "--date", "2020-01-01", "--hour", "10"])
    report_decel_descent_s = mock_report.call_args.kwargs["decel_descent_s"]

    assert search_decel_descent_s == report_decel_descent_s == DECEL_DESCENT_S
