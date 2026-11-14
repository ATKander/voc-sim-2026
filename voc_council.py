import json, re, statistics
import requests

MODEL = "qwen3:8b"
URL = "http://localhost:11434/api/chat"

HEEREN = ([("Amsterdam", f"merchant-am{i}", 0.5 + 0.1*i) for i in range(8)] +
          [("Zeeland", f"merchant-ze{i}", 0.5 + 0.1*i) for i in range(4)] +
          [("Delft", f"merchant-df{i}", 0.5 + 0.1*i) for i in range(2)] +
          [("Rotterdam", "merchant-rt", 0.5), ("Hoorn", "merchant-hn", 0.6),
           ("Enkhuizen", "merchant-en", 0.7)])

SYSTEM = ("You are a bewindhebber (director) of the VOC, an Amsterdam merchant "
          f"of the chamber of {{city}}, year {{year}}. Historical context: VOC has "
          "permanent capital of 6.42M guilders, trade brings 100-300% on voyages. "
          "Decide the annual policy. Respond ONLY with JSON.")

def ask_one(c, p, t, ctx, attempts=3):
    sys_text = SYSTEM.format(city=c, year=ctx["year"])
    rd = ctx.get("recent_divs")
    divs = "none" if rd in (None, "", []) else (
        ", ".join(f"{d}%" for d in rd) if isinstance(rd, (list, tuple)) else f"{rd}%")
    usr = (
        f"You are bewindhebber {p} of the chamber of {c}.\n"
        f"{ctx['retro']}\n"
        f"Year: {ctx['year']}. VOC treasury: {ctx['cash_mln']:.2f}M guilders. "
        f"Free capital above reserve: {ctx['cash_free_mln']:.2f}M. "
        f"Debt: {ctx['debt_mln']:.2f}M. Share price: {ctx['price']:.0f}.\n"
        f"Recent dividends: {divs}. Years since last dividend: {ctx['years_since_div']}. "
        f"Unrest: {ctx['unrest']:.2f}. War ongoing: {bool(ctx['war'])}.\n"
        f"Fleet at sea: {ctx['fleet_sea']}. Launched last year: {ctx['ships_last']}. "
        f"Revenue last year: {ctx['last_rev_mln']:.2f}M. "
        f"Credit remaining: {ctx['credit_left_mln']:.1f}M "
        f"(one ship ~{ctx['ship_cost_mln']:.2f}M, borrowed money costs 3-5%/yr).\n"
        "Steady-state guidance: a healthy VOC launches a STEADY 8-12 ships per "
        "year and pays a dividend every 5-10 years. Do not swing between "
        "extremes: alternating 2 and 16 ships between meetings is a policy "
        "failure, not caution. As a rule, change the launching rate by at most "
        "3-4 ships from the current level; large changes only when the treasury "
        "has collapsed or the fleet is far from strength. If debt approaches the "
        "credit limit, cut the launching rate - a company that cannot build or "
        "pay interest dies slowly. Dividends are paid only when they create no "
        "new debt; skipping them too long raises unrest (~+0.2/year, price "
        "collapses above unrest 3).\n"
        'Decide: div_rate (0-20), form ("cash" or "spices"), war (true/false), '
        "ships (1-20, within free capital plus remaining credit).\n"
        'Answer ONLY with JSON: {{"div_rate": N, "form": "...", "war": ..., "ships": N}}')
    for i in range(attempts):
        try:
            r = requests.post(URL, json={
                "model": MODEL, "think": False,
                "messages": [{"role": "system", "content": sys_text},
                             {"role": "user", "content": usr}],
                "options": {"temperature": t}, "stream": False}, timeout=120)
            raw = r.json()["message"]["content"]
            with open("council_raw.log", "a") as f:
                f.write(f"=== {ctx['year']} {c} {p} t={t} ===\n{raw[:500]}\n")
            m = re.search(r"\{.*\}", raw, re.S)
            v = json.loads(m.group(0))
            form = "spices" if "spice" in str(v["form"]).lower() else "cash"
            return dict(div_rate=float(min(20, max(0, v["div_rate"]))),
                        form=form, war=bool(v["war"]),
                        ships=int(min(20, max(1, v["ships"]))))
        except Exception as e:
            if i == 0: print("ask_one ошибка:", repr(e))
            continue
    return None

def assemble(ctx):
    votes = [v for v in (ask_one(c, p, t, ctx) for c, p, t in HEEREN) if v]
    if not votes:
        return dict(div_rate=0.0, form="spices", war=False, max_launch=10)
    half = len(votes) / 2
    return {"div_rate": statistics.median(v["div_rate"] for v in votes),
            "form": "spices" if sum(v["form"] == "spices" for v in votes) > half else "cash",
            "war": sum(v["war"] for v in votes) > half,
            "max_launch": statistics.median(v["ships"] for v in votes),
            "votes": votes}
