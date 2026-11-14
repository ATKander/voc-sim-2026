"""
Историко-материалистическая симуляция генезиса капитализма в Голландии (1585-1660).
Контрфактические флаги: disposs_on, centralization_on, permanent_capital, wisselbank.
"""
import numpy as np, pandas as pd

def gini_of(x):
    x = np.sort(np.maximum(0, x)); n = len(x)
    return float((2*np.arange(1, n+1) - n - 1).dot(x) / (n*x.sum())) if x.sum() else 0.0

BASE = dict(
    N=400, sigma=1.2,
    ship_cost=0.10e6, silver=0.12e6, markup=3.2,
    payroll_base=0.30e6, maint=0.012e6, asia_overhead=0.006e6,
    infra_growth=0.12e6, infra_cap=8.0e6, infra_rate=0.06,
    div_rate=7.0, div_reserve=0.8e6, div_form="cash",
    war=False, war_loss_extra=0.0, war_prize=2.5e6, war_sales_cut=0.20,
    fail_prob=0.03, fail_recover=0.97, centr=0.07, centr_unrest=0.05,
    wisselbank=True, disposs_on=True, centralization_on=True, permanent_capital=True,
    credit_limit=2.0e6, interest=0.05,
    jobs_k=0.9, cong=0.06, antw=0.004, dis_k=0.20, disposs=0.0015,
    scarcity=0.02, ship_loss=None, max_launch=16, a=0.35,
    pop_scale_nominal=True,
)
WAR_YEARS = set(range(1652, 1655)) | set(range(1665, 1668))
POP_REF = 50600.0

MODERATE = dict(div_rate=7.0, div_form="cash", war=False)
STINGY   = dict(div_rate=6.0, div_form="spices", war=False)
WARLIKE  = dict(div_rate=7.0, div_form="cash", war=True)

import voc_council
from das_loss import das


def run_dutch(p, years=range(1585, 1661), seed=42):
    rng = np.random.default_rng(seed)
    pop, W = 30000.0, 105.0
    wage_share = 0.35; state_rate = 8.33; prev_g = 0.03
    voc = False; nominal = 0.0; cash = 0.0; debt = 0.0; infra = 0.0
    cum_div = 0.0; voc_price = np.nan; unrest = 0.0
    fleet_out, fleet_mid, fleet_ret = [], [], []
    pool = 0.0; pop1600 = pop
    caps = rng.lognormal(6.5, p['sigma'], p['N'])
    caps = caps / caps.sum() * 3.67e6
    regent_idx = np.argsort(caps)[::-1][:17]
    records = []
    war_last = True; div_history = []; prev_rev = 0.0; last_div_year = None
    prev_snap = None; last_mandate = None
    council_pol = dict(div_rate=0.0, form="spices", war=False, max_launch=10)

    for year in years:
        ret = 0.060 if year <= 1620 else (0.045 if year <= 1650 else 0.032)
        if p['wisselbank'] and year >= 1609: ret += 0.004
        if year in WAR_YEARS: ret -= 0.010

        if year == 1602:
            voc = True
            scale = (pop1600 / POP_REF) if p.get('pop_scale_nominal') else 1.0
            nominal = 6.42e6 * scale
            cash = nominal; pool = nominal

        jobs = float(np.clip(0.006 + 0.55*ret, 0.004, 0.028))
        c0 = 0.012 if year <= 1620 else (0.001 if year <= 1650 else -0.0095)
        drag = p['scarcity'] * max(0.0, 0.50 - wage_share)
        r_real = float(np.clip(c0 + p['a']*(jobs - prev_g) + drag, -0.010, 0.014))
        W *= (1 + r_real)
        add = 0.004 + p['jobs_k']*jobs + (p['antw'] if year <= 1600 else 0.0)
        if p['disposs_on']: add += p['dis_k']*(wage_share - 0.35)
        prev_g = float(np.clip(add - p['cong']*(pop/200000.0)**2, 0.002, 0.045))
        pop *= (1 + prev_g)
        if year == 1600: pop1600 = pop
        if p['disposs_on']:
            wage_share = min(0.60, wage_share + p['disposs'] + (0.002 if 1607 <= year <= 1640 else 0.0))
        if year == 1640: state_rate = 5.0

        revenue = 0.0; ships_launched = 0
        if voc:
            if year == 1602 or (year - 1602) % 5 == 0:
                if prev_snap is None:
                    retro = ("This is the first meeting; there is no previous "
                             "mandate to review.")
                else:
                    fl_now = len(fleet_out) + len(fleet_mid) + len(fleet_ret)
                    retro = (f"Review of your last {year - prev_snap['at']}-year mandate "
                             f"({prev_snap['mandate']} ships/yr): fleet {prev_snap['fleet']}->{fl_now}, "
                             f"revenue {prev_snap['rev']/1e6:.2f}M->{prev_rev/1e6:.2f}M per year, "
                             f"debt {prev_snap['debt']/1e6:.2f}M->{debt/1e6:.2f}M, "
                             f"unrest {prev_snap['unrest']:.2f}->{unrest:.2f}, "
                             f"cumulative dividends {prev_snap['cum_div']:.1f}%->{cum_div:.1f}%. "
                         f"Actual outbound loss rate (DAS): {das.loss_rate(year)*100:.1f}%.")
                ctx = dict(year=year, cash_mln=cash/1e6, debt_mln=debt/1e6,
                           price=100.0 if np.isnan(voc_price) else float(voc_price),
                           recent_divs=div_history[-3:], unrest=unrest, war=war_last,
                           fleet_sea=len(fleet_out) + len(fleet_mid) + len(fleet_ret),
                           last_rev_mln=prev_rev/1e6,
                           cash_free_mln=max(0.0, (cash - p['div_reserve'])/1e6),
                           ship_cost_mln=p['ship_cost']/1e6,
                           years_since_div=(year - last_div_year) if last_div_year else (year - 1602 + 1),
                           credit_left_mln=max(0.0, (p['credit_limit'] - debt)/1e6),
                           retro=retro,
                           ships_last=records[-1]["ships"] if records else 0)
                council_pol = voc_council.assemble(ctx)
                if (council_pol["div_rate"] <= 0
                        and (year - (last_div_year if last_div_year else 1601)) >= 5
                        and (cash - p['div_reserve']) > 0.25 * nominal):
                    council_pol["div_rate"] = 2.5
                    council_pol["form"] = "cash"
                    print(f"{year}: ревизия участников вынуждает выплату 2.5% наличными")
                if council_pol["div_rate"] > 0:
                    div_history.append(council_pol["div_rate"])
                    last_div_year = year
                prev_snap = dict(fleet=len(fleet_out) + len(fleet_mid) + len(fleet_ret),
                                 rev=prev_rev, debt=debt, unrest=unrest,
                                 cum_div=cum_div, mandate=last_mandate, at=year)
                war_last = council_pol["war"]
                last_mandate = council_pol["max_launch"]
            war = council_pol["war"] and year >= 1605
            for i in range(len(fleet_out)):
                purse = cash if p['permanent_capital'] else pool
                if purse + p['credit_limit'] >= p['silver']:
                    if p['permanent_capital']: cash -= p['silver']
                    else: pool -= p['silver']
                    fleet_mid.append(p['silver'])
                else:
                    fleet_mid.append(0.0)
            fleet_out = []
            for gc in fleet_mid: fleet_ret.append(gc)
            fleet_mid = []
            for gc in list(fleet_ret):
                loss_p = das.loss_rate(year) if p['ship_loss'] is None else p['ship_loss']
                if rng.random() < loss_p:
                    fleet_ret.remove(gc); continue
                rev = gc * p['markup'] * (1 - p['war_sales_cut'] if war else 1.0)
                if p['permanent_capital']: cash += rev
                else: caps *= (1 + rev / caps.sum())
                revenue += rev
                if war: cash += p['war_prize'] / 16.0
            fleet_ret = []
            if p['permanent_capital']:
                budget = cash - p['div_reserve'] + max(0.0, p['credit_limit'] - debt)
            else:
                budget = pool; pool *= 0.96
            while ships_launched < int(council_pol["max_launch"]) and budget >= p['ship_cost']:
                if p['permanent_capital']: cash -= p['ship_cost']
                else: pool -= p['ship_cost']
                budget -= p['ship_cost']; ships_launched += 1
            fleet_out = [1]*ships_launched
            fleet_sea = 3 * ships_launched
            payroll = p['payroll_base'] * (W / 155.0)
            costs = payroll + p['maint']*fleet_sea + p['asia_overhead']*fleet_sea
            infra = min(p['infra_cap'], infra + (p['infra_growth'] if year >= 1605 else 0.0))
            costs += p['infra_rate'] * infra
            if p['permanent_capital']: cash -= costs
            else: pool -= min(pool, costs)
            if cash < 0: debt += -cash; cash = 0.0
            if debt > 0:
                interest = min(cash, debt * p['interest']); cash -= interest; debt -= interest
            if p['permanent_capital']:
                if council_pol["div_rate"] > 0 and cash > p['div_reserve']:
                    amount = min(cash - p['div_reserve'], council_pol["div_rate"]/100 * nominal)
                    cash -= amount
                    eff = amount * (0.77 if council_pol["form"] == "spices" else 1.0)
                    caps *= (1 + eff / caps.sum()); cum_div += eff / nominal * 100
                    unrest *= 0.5 if p['div_form'] == "cash" else 0.9
                else:
                    unrest += 0.2
            else:
                if pool < 0.5*nominal/21: unrest += 0.1
            if p['centralization_on']:
                mid_m = (caps >= 1000) & (caps < 20000)
                fail = (rng.random(len(caps)) < p['fail_prob']) & mid_m
                if fail.any():
                    caps[int(np.argmax(caps))] += caps[fail].sum() * p['fail_recover']
                    caps[fail] = 0.0
                f = p['centr'] + p['centr_unrest'] * min(unrest, 3)
                if 1636 <= year <= 1639: f += 0.20
                small = caps < 1000
                if small.any():
                    lot = caps[small].sum() * f; caps[small] *= (1 - f)
                    caps[int(np.argmax(caps))] += lot
            if p['permanent_capital']:
                voc_price = 100 + 0.65*cum_div - 30*min(unrest, 3)
                if 1636 <= year <= 1638: voc_price *= 1.15
                if year == 1639: voc_price /= 1.12
            else:
                voc_price = 100.0

        prev_rev = revenue
        reg_share = 100*caps[regent_idx].sum()/caps.sum() if voc else np.nan
        records.append(dict(year=year, pop=round(pop/1000, 1), wage_index=round(W, 1),
                            wage_share=round(100*wage_share, 1),
                            voc_price=round(voc_price, 1) if voc else np.nan,
                            state_rate=state_rate, gini=round(gini_of(caps), 3),
                            cum_div=round(cum_div, 1), cash_mln=round(cash/1e6, 2),
                            debt_mln=round(debt/1e6, 2), ships=ships_launched,
                            unrest=round(unrest, 2),
                            regent_share=round(reg_share, 1) if voc else np.nan))
    return pd.DataFrame(records)

if __name__ == "__main__":
    for name, c in [("base", dict(BASE, **MODERATE)), ("stingy", dict(BASE, **STINGY)),
                    ("warlike", dict(BASE, **WARLIKE)),
                    ("cf1_no_dispossession", dict(BASE, **MODERATE, disposs_on=False)),
                    ("cf2_no_centralization", dict(BASE, **MODERATE, centralization_on=False)),
                    ("cf3_no_permanent_capital", dict(BASE, **MODERATE, permanent_capital=False)),
                    ("cf4_no_wisselbank", dict(BASE, **MODERATE, wisselbank=False))]:
        run_dutch(c).to_csv(f"metrics_{name}.csv", index=False)
