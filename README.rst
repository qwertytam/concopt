concopt
=======

Optimise the flight time of the FS Labs Concorde (Prepar3D v5) on JFK to LHR.
Two tools:

* **Day search** -- scans every historical departure Active Sky can reproduce
  (08:00-14:00 New York local, August 2014 to today, about 31,000 candidates)
  against ERA5 reanalysis winds and temperatures, and ranks them by total
  block time, including fuel-driven takeoff weight, the climb and arrival
  models, and a runway wind screen.
* **In-flight advisor** -- reads the live sim over SimConnect and Active Sky,
  recommends the best cruise level ahead, and can record the flight for
  comparison against the plan.

Status
------

A personal project, used a handful of times: minimum viable, not packaged
for anyone else. Both tools work end to end. The performance tables come from
the Air France Concorde manual figures in ``src/concopt/data/``.

Install
-------

Python 3.11+ and Poetry::

    poetry install --extras test
    poetry run pytest

Commands
--------

``poetry run concopt <command>``:

==============  ==========================================================
``route``       parse a ``.pln`` and list its legs
``search``      rank every candidate departure
``shortlist``   turn the top of a search into ready-to-run ``verify`` lines
``verify``      check one candidate against Active Sky
``report``      full per-leg plan for one day and time
``inflight``    live advisor and flight recorder
==============  ==========================================================

``notebooks/day-search-results.ipynb`` plots a search run and the winning
day's full flight (profile, limits, fuel, phases, route map).

Documentation
-------------

* `RUNBOOK <docs/RUNBOOK.md>`_ -- how to run it, start to finish: download and
  reduce the weather, search, verify, report, fly, analyse.
* `METHOD <METHOD.md>`_ -- why the search works the way it does.
* `Implementation plan <docs/IMPLEMENTATION-PLAN.md>`_ -- project status,
  decisions and known gaps.
