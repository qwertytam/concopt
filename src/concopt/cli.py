"""concopt command-line entry point. argparse-based subcommand dispatcher;
`route` is phase 3, `search` and `inflight` land here in later phases."""
import argparse

from concopt.route import build_legs, parse_pln, supersonic_segment


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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
