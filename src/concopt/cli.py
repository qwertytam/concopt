"""concopt command-line entry point. argparse-based subcommand dispatcher;
`route` is phase 3, `search`/`report` are phase 3b, `inflight` lands here in
a later phase."""
import argparse
import datetime as dt

from concopt.limits import CRUISE_MACH
from concopt.report import DEFAULT_DECEL_DESCENT_S, best_candidate_from_csv, run_report
from concopt.route import build_legs, parse_pln, supersonic_segment
from concopt.search import DEPARTURE_TO_ACCEL_S, run_search


def _add_common_route_args(parser):
    """--pln/--accel/--decel, shared verbatim by every subcommand that
    works from a parsed flight plan."""
    parser.add_argument('--pln', required=True, help='path to a P3D .pln flight plan')
    parser.add_argument('--accel', default='LINND',
                         help='acceleration waypoint id (default: LINND)')
    parser.add_argument('--decel', default='BARIX',
                         help='deceleration waypoint id (default: BARIX)')


def _cmd_route(args):
    plan = parse_pln(args.pln)
    legs = build_legs(plan['waypoints'], max_leg_nm=args.max_leg_nm)
    mask = supersonic_segment(legs, accel_id=args.accel, decel_id=args.decel)

    print(f"{'idx':>3}  {'from_id':<8}{'to_id':<8}{'lat_mid':>10}{'lon_mid':>11}"
          f"{'track':>8}{'leg_nm':>9}{'cum_nm':>10}  ss")
    for i, (leg, ss) in enumerate(zip(legs, mask)):
        print(f"{i:>3}  {leg.from_id:<8}{leg.to_id:<8}{leg.lat_mid:>10.4f}{leg.lon_mid:>11.4f}"
              f"{leg.track_deg:>8.1f}{leg.dist_nm:>9.1f}{leg.cum_nm:>10.1f}  {'*' if ss else ''}")

    total_nm = legs[-1].cum_nm if legs else 0.0
    ss_nm = sum(leg.dist_nm for leg, ss in zip(legs, mask) if ss)
    pct = (ss_nm / total_nm * 100.0) if total_nm else 0.0
    print(f"\n{len(legs)} legs, {total_nm:.1f} nm total, "
          f"{ss_nm:.1f} nm supersonic ({pct:.0f}%)")


def _cmd_search(args):
    run_search(args.pln, args.npz, accel_id=args.accel, decel_id=args.decel,
               top=args.top, out_path=args.out,
               departure_to_accel_s=args.departure_to_accel_min * 60.0,
               cruise_mach=args.cruise_mach)


def _cmd_report(args):
    if args.best:
        local_date, local_hour = best_candidate_from_csv(args.search_csv)
    elif args.date and args.hour is not None:
        local_date, local_hour = dt.date.fromisoformat(args.date), args.hour
    else:
        raise SystemExit('report: either --best, or both --date and --hour, is required')

    run_report(args.pln, args.npz, local_date, local_hour,
               accel_id=args.accel, decel_id=args.decel, out_path=args.out,
               departure_to_accel_s=args.departure_to_accel_min * 60.0,
               decel_descent_s=args.decel_descent_min * 60.0,
               cruise_mach=args.cruise_mach)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    _add_common_route_args(common)

    parser = argparse.ArgumentParser(prog='concopt')
    subparsers = parser.add_subparsers(dest='command', required=True)

    route_parser = subparsers.add_parser(
        'route', parents=[common], help='parse a .pln and list its legs')
    route_parser.add_argument('--max-leg-nm', type=float, default=100.0,
                               help='subdivide legs longer than this (nm, default: 100)')
    route_parser.set_defaults(func=_cmd_route)

    search_parser = subparsers.add_parser(
        'search', parents=[common], help='rank candidate departures by supersonic-segment time')
    search_parser.add_argument('--npz', required=True, help='path to the .npz from era5.reduce_to_legs')
    search_parser.add_argument('--top', type=int, default=50,
                                help='number of ranked rows to write out (default: 50)')
    search_parser.add_argument('--out', default='results.csv', help='output CSV path (default: results.csv)')
    search_parser.add_argument('--departure-to-accel-min', type=float,
                                default=DEPARTURE_TO_ACCEL_S / 60.0,
                                help='minutes from brakes release to the accel '
                                     'point/first supersonic leg (default: 20)')
    search_parser.add_argument('--cruise-mach', type=float, default=CRUISE_MACH,
                                help='target cruise Mach used in place of Mmo '
                                     f'(default: {CRUISE_MACH}; try 2.04 for Mmo)')
    search_parser.set_defaults(func=_cmd_search)

    report_parser = subparsers.add_parser(
        'report', parents=[common], help='full per-leg breakdown for one candidate departure')
    report_parser.add_argument('--npz', required=True, help='path to the .npz from era5.reduce_to_legs')
    report_parser.add_argument('--date', help='local (America/New_York) departure date, YYYY-MM-DD')
    report_parser.add_argument('--hour', type=int, help='local departure hour, 24h (08-14)')
    report_parser.add_argument('--best', action='store_true',
                                help='use the fastest candidate from --search-csv instead of --date/--hour')
    report_parser.add_argument('--search-csv', default='results.csv',
                                help='concopt search --out CSV to read --best from (default: results.csv)')
    report_parser.add_argument('--out', default='report.csv', help='output CSV path (default: report.csv)')
    report_parser.add_argument('--departure-to-accel-min', type=float,
                                default=DEPARTURE_TO_ACCEL_S / 60.0,
                                help='minutes from brakes release to the accel '
                                     'point/first supersonic leg (default: 20)')
    report_parser.add_argument('--decel-descent-min', type=float,
                                default=DEFAULT_DECEL_DESCENT_S / 60.0,
                                help='minutes from the decel point to touchdown, decel + descent '
                                     '(default: seeded from conc_desc_time.csv at FL600)')
    report_parser.add_argument('--cruise-mach', type=float, default=CRUISE_MACH,
                                help='target cruise Mach used in place of Mmo '
                                     f'(default: {CRUISE_MACH}; try 2.04 for Mmo)')
    report_parser.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
