"""Tests of core/physicality.py, each with a hand-worked answer.

Several are named after bugs the module once had. Run with pytest or `python tests/test_physicality.py`.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
import sympy as sp                                                     # noqa: E402

import Empirical_DF_Construction.core.physicality as ph                # noqa: E402
from Empirical_DF_Construction.core.physicality import (               # noqa: E402
    DECADES, Verdict, as_expr, check, dps_for, generate, half_grid,
    low_energy_limit, _capped, _LimitTimeout, _termwise_limit)

L = "log(eps_hat/(1 - eps_hat))"


@contextmanager
def _stalled(name: str, seconds: int = 30):
    """Run `check` with one of its phases replaced by a call that outlasts its budget."""
    real = getattr(ph, name)

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return real(*args, **kwargs)

    setattr(ph, name, slow)
    try:
        yield
    finally:
        setattr(ph, name, real)


# --------------------------------------------------------------------------------------
# the arithmetic the module is built around
# --------------------------------------------------------------------------------------

def test_float64_cannot_write_down_the_region_being_swept():
    assert 1.0 - 1e-18 == 1.0
    assert np.longdouble(1) - np.longdouble(10) ** -20 == np.longdouble(1)


def test_the_t_parameterisation_reaches_where_eps_hat_cannot():
    """t = 1 - eps_hat is a small number, which is what floating point is good at."""
    g = half_grid()
    assert g.min() <= 10.0 ** -(DECADES - 1)
    assert (g > 0).all() and (g <= 0.5).all()
    assert np.isfinite(g).all()


def test_confirmation_precision_follows_the_abscissa():
    """A fixed budget cannot resolve 1 - t for arbitrarily small t; this one scales."""
    assert dps_for(1e-300) > 300
    assert dps_for(0.3) == dps_for(0.5)                 # the floor, in the bulk
    assert dps_for(0.0) == dps_for(0.4)                 # degenerate input, no crash


def test_both_halves_are_generated_and_carry_the_same_derivative_convention():
    """dy/deps_hat is taken in eps_hat on both sides, so the sign needs no reinterpreting."""
    lo, hi = generate(as_expr(f"{L} - 2.3098297"))
    assert lo.name == "lower" and hi.name == "upper"
    assert np.nanmin(lo.dy) > 0 and np.nanmin(hi.dy) > 0


# --------------------------------------------------------------------------------------
# monotonicity
# --------------------------------------------------------------------------------------

def test_logit_is_increasing():
    assert check(f"{L} - 2.3098297").monotone is Verdict.NOT_REFUTED


def test_decreasing_form_is_refuted():
    assert check(f"-{L}").monotone is Verdict.REFUTED


def test_constant_is_not_strictly_increasing():
    r = check("-2.19475410000000")
    assert r.monotone is Verdict.REFUTED
    assert r.refuted


def test_a_form_that_saturates_flat_is_not_refuted_for_being_flat():
    """From fold 1's centered_ratio front, and the reason _pair_falls compares strictly."""
    r = check("4.1966376 - 6.41859*exp(-0.048954766*(eps_hat - 1/2)"
              "/(eps_hat*(1 - eps_hat)))")
    assert r.monotone is Verdict.NOT_REFUTED
    assert not r.refuted


def test_a_genuinely_constant_form_is_still_caught_by_the_derivative():
    r = check("-2.19475410000000")
    assert r.monotone is Verdict.REFUTED
    assert "dy/deps_hat" in r.detail["monotone"]        # refuted there, not on the values


def test_turnover_near_the_upper_boundary_is_caught():
    """Falls over only as eps_hat -> 1, which the fitted grid would never reach."""
    assert check(f"{L} - 10*eps_hat/(1 - eps_hat)").monotone is Verdict.REFUTED


# --------------------------------------------------------------------------------------
# well-definedness
# --------------------------------------------------------------------------------------

def test_offset_inside_the_log_puts_a_branch_point_in_the_domain():
    """log(eps/(1-eps) - d) is undefined below eps_hat = d/(1+d), inside (0, 1)."""
    r = check("log(eps_hat/(1 - eps_hat) - 0.010945489) - 2.27")
    assert r.well_defined is Verdict.REFUTED


def test_a_pole_the_fitted_grid_could_never_see_is_caught():
    e = f"{L} - 2.0 - 0.1/(34.538776394910684 - {L})"
    assert check(e).refuted


def test_interior_pole_is_refuted():
    assert check(f"{L} - 1.87 - {L}/(12.331261 - {L})").refuted


def test_well_behaved_form_is_not_refuted():
    r = check(f"{L} - 2.264420642 - 0.012469158/eps_hat")
    assert not r.refuted


# --------------------------------------------------------------------------------------
# low-energy: swept for evidence, decided by the symbolic limit
# --------------------------------------------------------------------------------------

def test_the_slow_divergence_is_never_refuted():
    """-log(-log(eps_hat)) -> -inf, is admissible, and is the reason no *sweep* decides the
    low-energy condition."""
    r = check("-log(-log(eps_hat))")
    assert r.monotone is Verdict.NOT_REFUTED
    assert r.well_defined is Verdict.NOT_REFUTED
    assert not r.refuted


def test_the_limit_proves_what_the_sweep_could_only_decline_to_refute():
    v, detail = low_energy_limit(as_expr("-log(-log(eps_hat))"))
    assert v is Verdict.HOLDS and detail == "-oo"
    assert check("-log(-log(eps_hat))").low_energy is Verdict.HOLDS


def test_a_form_that_levels_off_is_refuted_on_low_energy():
    r = check("sinh(2*eps_hat - 2.4128523)")
    assert r.monotone is Verdict.NOT_REFUTED
    assert r.well_defined is Verdict.NOT_REFUTED
    assert r.low_energy is Verdict.REFUTED
    assert r.refuted                       # and refuted overall, on that count alone


def test_a_diverging_but_wrong_signed_limit_is_refuted():
    """+oo is not -oo. A form running away upward at the escape energy fails too."""
    assert check(f"-{L}").low_energy is Verdict.REFUTED


@contextmanager
def _slow_limit(stall_on):
    """Make sympy's limit stall (sleep past any budget) on the expressions `stall_on` picks."""
    real = ph.sp.limit

    def limit(e, *args, **kwargs):
        if stall_on(e):
            time.sleep(30)
        return real(e, *args, **kwargs)

    ph.sp.limit = limit
    try:
        yield
    finally:
        ph.sp.limit = real


def test_a_limit_that_times_out_is_undecided_and_not_a_pass():
    e = as_expr("log(sinh(eps_hat))*sinh(asin(eps_hat) + 0.8560545)")
    with _slow_limit(lambda x: True):
        v, detail = low_energy_limit(e, timeout=2)
    assert v is Verdict.UNDECIDED
    assert "2s" in detail
    assert v is not Verdict.HOLDS


def test_a_hard_sum_is_rescued_by_the_limits_of_its_terms():
    """fold 1's c = 11 row: whole, its limit ran past 300 s on the cluster's sympy; its terms give
    -3.3564768, -oo and sinh(0.8560545), and finite plus -oo is -oo."""
    e = as_expr("log(sinh(eps_hat)) + sinh(asin(eps_hat) + 0.8560545) - 3.3564768")
    with _slow_limit(lambda x: x == e):
        v, detail = low_energy_limit(e, timeout=3)
    assert v is Verdict.HOLDS
    assert "terms" in detail


def test_the_termwise_fallback_refuses_an_indeterminate_sum():
    """+oo and -oo in the same sum is genuinely indeterminate and no termwise work settles it, so
    the fallback must decline rather than pick one."""
    assert _termwise_limit(as_expr("-log(eps_hat) + log(sinh(eps_hat)) + 1.5"), 30) is None


def test_the_termwise_fallback_settles_a_sum_it_can():
    """The other side of the same guard: exactly one infinite term, so the sum follows."""
    e = as_expr("log(eps_hat) + sinh(asin(eps_hat) + 0.8560545) - 3.3564768")
    assert _termwise_limit(e, 30) == -sp.oo


def test_the_timeout_cannot_be_swallowed_by_sympy():
    """It derives from BaseException on purpose: sympy catches Exception broadly, and a timeout it
    can catch is a timeout that does not fire."""
    from Empirical_DF_Construction.core.physicality import _LimitTimeout
    assert issubclass(_LimitTimeout, BaseException)
    assert not issubclass(_LimitTimeout, Exception)


def test_the_ladder_and_the_limit_disagree_and_the_limit_is_the_verdict():
    r = check("-log(-log(eps_hat))")
    assert float(dict(r.ladder)["1e-30"]) > -5        # what a reader of the ladder sees
    assert r.low_energy is Verdict.HOLDS              # what is actually true


def test_the_ladder_shows_why_a_trend_cannot_decide_it():
    """Side by side: one of these vanishes and one does not, and over thirty decades the ladder of
    the first barely moves."""
    slow = dict(check("-log(-log(eps_hat))").ladder)
    fast = dict(check(f"{L} - 2.3098297").ladder)
    assert float(slow["1e-12"]) > -4 and float(slow["1e-30"]) > -5
    assert float(fast["1e-12"]) < -29
    for d in (slow, fast):                       # both really are still falling
        assert float(d["1e-30"]) < float(d["1e-12"])


# --------------------------------------------------------------------------------------
# the machinery
# --------------------------------------------------------------------------------------

def test_a_row_can_be_given_in_u_with_its_transform():
    """pysr_composed.csv records equation_u beside the transform; composing here tests what PySR
    fitted rather than what a simplifier made of it."""
    a = check("u - 2.3098297", transform="logit")
    b = check(f"{L} - 2.3098297")
    assert (a.monotone, a.well_defined) == (b.monotone, b.well_defined)


def test_unparseable_input_is_refused_not_screened():
    r = check("log(((")
    assert r.refuted and "parse" in r.detail


def test_unknown_transform_is_refused():
    assert check("u - 1", transform="not_a_transform").refuted


def test_verdicts_are_stable_across_repeated_calls():
    e = f"{L} - 1.87 - {L}/(12.331261 - {L})"
    a, b = check(e), check(e)
    assert (a.monotone, a.well_defined) == (b.monotone, b.well_defined)


def test_a_denser_grid_does_not_change_a_settled_verdict():
    """Density is what buys confidence in a sweep, so a verdict that moves with it was not a verdict."""
    dense = half_grid(per_decade=128)
    for e in (f"-{L}", "-2.19475410000000", f"{L} - 10*eps_hat/(1 - eps_hat)"):
        assert check(e).monotone is Verdict.REFUTED
        assert check(e, grid=dense).monotone is Verdict.REFUTED


# --------------------------------------------------------------------------------------
# the caps on the numeric phases, and what a cap is allowed to conclude
# --------------------------------------------------------------------------------------

def test_a_numeric_phase_that_times_out_is_undecided_and_not_refuted():
    """A cap that fired is a question unanswered, not a counterexample found."""
    with _stalled("check_monotone"):
        r = check(f"{L} - 2.3", test_timeout=2)
    assert r.monotone is Verdict.UNDECIDED
    assert r.monotone is not Verdict.REFUTED
    assert "monotone" in r.detail["timed_out_on"]


def test_a_timed_out_form_does_not_survive():
    """The trap: UNDECIDED is not REFUTED, so a timeout used to pass straight through."""
    e = f"{L} - 2.3"
    clean = check(e, test_timeout=300)
    assert clean.survives and clean.status == "survives"

    with _stalled("check_monotone"):
        r = check(e, test_timeout=2)
    assert not r.refuted, "nothing refuted it -- that is the whole point of the case"
    assert not r.survives, "a form that was never checked has not survived anything"
    assert r.status == "undecided"


def test_a_refutation_outranks_a_timeout_on_another_condition():
    """A counterexample is positive evidence and does not stop being evidence because a different
    condition ran out of time."""
    with _stalled("check_monotone"):
        r = check("-6.3070908*exp(-sqrt(eps_hat)/sqrt(1 - eps_hat))", test_timeout=2)
    assert r.monotone is Verdict.UNDECIDED
    assert r.low_energy is Verdict.REFUTED
    assert r.refuted and not r.survives
    assert r.status == "refuted"


def test_a_timeout_building_the_sides_leaves_both_numeric_conditions_undecided():
    """The sides are the shared cost, so their cap is the one that takes out two phases."""
    with _stalled("generate"):
        r = check(f"{L} - 2.3", test_timeout=2)
    assert r.monotone is Verdict.UNDECIDED
    assert r.well_defined is Verdict.UNDECIDED
    assert r.detail["timed_out_on"] == "generate"
    assert not r.survives and r.status == "undecided"


def test_the_cap_gives_the_previous_handler_back():
    """`low_energy_limit` arms its own SIGALRM after the numeric phases have disarmed theirs, so a
    cap that leaked its handler would change what the next phase does."""
    import signal
    before = signal.getsignal(signal.SIGALRM)
    with _capped(30):
        pass
    assert signal.getsignal(signal.SIGALRM) is before


def test_the_cap_interrupts_and_is_not_swallowed():
    """Same reason the limit timeout derives from BaseException: a broad `except Exception` inside
    a phase must not be able to catch the cap and carry on past it."""
    fired = False
    try:
        with _capped(1):
            try:
                time.sleep(10)
            except Exception:                    # noqa: BLE001 - deliberately broad
                raise AssertionError("the cap was swallowed by an except Exception")
    except _LimitTimeout:
        fired = True
    assert fired


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            bad += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            bad += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
