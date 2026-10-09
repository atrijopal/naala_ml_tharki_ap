"""End-to-end test of the mobile UI in real (headless) Firefox, with screenshots. Not part of tests/test_core.py.
  DEMO=1 uvicorn splitsnap.app:app --port 8765 &      then:      python tests/e2e_ui.py [--dark] [--wide] [--out out/ui]
Needs: selenium, and the snap Firefox + geckodriver paths below (override with FIREFOX_BIN / GECKODRIVER)."""
import argparse, os, sys, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ap = argparse.ArgumentParser()
ap.add_argument("--dark", action="store_true"); ap.add_argument("--wide", action="store_true")
ap.add_argument("--out", default="out/ui"); ap.add_argument("--url", default="http://localhost:8765/")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
os.environ["TMPDIR"] = os.path.expanduser("~/ffprofile-tmp"); os.makedirs(os.environ["TMPDIR"], exist_ok=True)
tag = ("dark" if a.dark else "light") + ("-wide" if a.wide else "-phone")

o = Options(); o.add_argument("-headless")
o.binary_location = os.environ.get("FIREFOX_BIN", "/snap/firefox/current/usr/lib/firefox/firefox")
o.set_preference("layout.css.prefers-color-scheme.content-override", 0 if a.dark else 1)
d = webdriver.Firefox(service=Service(os.environ.get("GECKODRIVER", "/snap/bin/geckodriver")), options=o)
d.set_window_size(*((1280, 900) if a.wide else (390, 900)))
W = WebDriverWait(d, 10)
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def shot(name):
    time.sleep(0.4); d.get_full_page_screenshot_as_file(f"{a.out}/{tag}-{name}.png")


def css(sel): return W.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel)))
def click(sel):
    e = W.until(EC.presence_of_element_located((By.CSS_SELECTOR, sel))); d.execute_script("arguments[0].scrollIntoView({block:'center'})", e)
    W.until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel))).click()
def many(sel): return d.find_elements(By.CSS_SELECTOR, sel)
def text(sel): return css(sel).text
def settle(): time.sleep(0.7)                                   # validation is debounced 250 ms


def set_value(sel, value):
    e = css(sel); d.execute_script("arguments[0].scrollIntoView({block:'center'})", e); e.click(); e.send_keys(Keys.CONTROL, "a"); e.send_keys(Keys.DELETE); e.send_keys(value)


def money(s): return float(s.replace("₹", "").replace(",", "").replace("−", "-").strip())


try:
    d.get(a.url); css("h1")
    check(text("h1") == "Add the bill", "step 1 shows 'Add the bill'")
    check("null" not in d.find_element(By.ID, "main").text and "false" not in d.find_element(By.ID, "main").text, "no stray null/false text")
    shot("1-photo")

    click("#sample2"); W.until(lambda x: text("h1") == "Check what we read"); settle()
    check(len(many(".field.low")) >= 3, f"low-confidence fields are highlighted ({len(many('.field.low'))})")
    check(len(many(".field.bad")) >= 1, f"arithmetic problems are marked ({len(many('.field.bad'))})")
    check("doesn't add up" in text("#tick"), "the total check says it doesn't add up")
    shot("2-check-messy")
    fixbtn = many(".note .fix")
    check(len(fixbtn) == 1 and "65" in fixbtn[0].text, "a one-tap fix is offered for the misread rate: " + (fixbtn[0].text if fixbtn else "none"))
    fixbtn[0].click(); settle()
    check(css('input[data-path="items[2].unit_price"]').get_attribute("value") == "65", "tapping the fix sets the rate to 65")
    check(not many('[data-fpath="items[2].total"].bad'), "the arithmetic mark is gone after the fix")
    set_value('input[data-path="items[2].unit_price"]', "60"); settle()

    set_value('input[data-path="items[2].total"]', "120"); settle()
    check(not many('[data-fpath="items[2].total"].low'), "editing a low-confidence field clears its highlight")
    set_value('input[data-path="subtotal"]', "760"); set_value('input[data-path="grand_total"]', "797.35"); settle()
    check(len(many(".field.bad")) == 0, "after the corrections nothing is marked bad")
    check("Adds up" in text("#tick"), "the check turns to 'Adds up'")
    check("797.35" in text("#bar .tot"), "the bar total follows the edit: " + text("#bar .tot"))
    shot("2-check-fixed")
    click("#add-item"); n = len(many(".item")); check(n == 5, "an item row can be added"); d.execute_script("arguments[0].scrollIntoView({block:'center'})", many(".item .danger")[-1]); many(".item .danger")[-1].click()
    check(len(many(".item")) == 4, "an item row can be removed")
    check(any(l.text.startswith("Items") for l in many("#ledger .row")), "the charge ledger is shown")

    click("#to-people"); W.until(lambda x: text("h1") == "Who's eating")
    for n in ("Riya", "Aman", "Sara"):
        e = css("#pname"); e.send_keys(n, Keys.ENTER); time.sleep(0.15)
    e = css("#pname"); e.send_keys("riya", Keys.ENTER); time.sleep(0.2)
    check("already" in text(".note.bad"), "a duplicate name is refused (case-insensitive)")
    check(len(many(".people li")) == 3, "three people added")
    shot("3-people")

    click("#to-assign"); W.until(lambda x: text("h1") == "Who had what")
    check(many("#to-split")[0].get_attribute("aria-disabled") == "true", "the split button waits until every item is assigned")
    check("4 items left" in text("#bar .sub"), "the bar says how many items are left: " + text("#bar .sub"))
    click("#equal-all"); time.sleep(0.2)
    for who in ("Aman", "Sara"):                                   # the page redraws after each tap: look the chip up fresh
        c = many(".assign-item")[0].find_element(By.CSS_SELECTOR, f'.chip[data-person="{who}"]')
        d.execute_script("arguments[0].scrollIntoView({block:'center'})", c); c.click(); time.sleep(0.2)
    item0 = many(".assign-item")[0]
    on = [c.get_attribute("data-person") for c in item0.find_elements(By.CSS_SELECTOR, '.chip[aria-pressed="true"]')]
    check(on == ["Riya"], f"item 1 is only Riya's: {on}")
    plus = many(".assign-item")[3].find_elements(By.CSS_SELECTOR, '.stepper button[aria-label^="More shares for Sara"]')[0]; d.execute_script("arguments[0].scrollIntoView({block:'center'})", plus); plus.click(); time.sleep(0.2)
    check(many("#to-split")[0].get_attribute("aria-disabled") == "false", "the split button is ready")
    shot("4-assign")

    click("#to-split"); W.until(lambda x: len(many(".person")) == 3); time.sleep(0.5)
    big = money(text(".big-total")); finals = [money(e.text) for e in many(".person .head .amt")]
    check(abs(sum(finals) - big) < 0.005, f"the three amounts add up to the bill: {finals} = {sum(finals):.2f} vs {big:.2f}")
    check(abs(big - 797.35) < 0.005, f"the bill total is the corrected one: {big}")
    rows = [[c.text for c in r.find_elements(By.CSS_SELECTOR, "th,td")] for r in many("#matrix tr")]
    check(len(rows) >= 4 and rows[0][-1] == "Bill", "the charges-by-people table has a Bill column")
    ok = all(abs(sum(money(c) for c in r[1:-1]) - money(r[-1])) < 0.015 for r in rows[1:])
    check(ok, "every row of the charges-by-people table adds up to its Bill figure")
    check(abs(money(rows[-1][-1]) - big) < 0.005, "the table's last row equals the bill total")
    check(len(many(".sharebar .seg")) == 3, "the share bar has one segment per person")
    many(".person details summary")[0].click(); time.sleep(0.3)
    check("Riya" in many(".person")[0].text and "full" in many(".person")[0].text and "Your items" in many(".person")[0].text, "per-person breakdown lists the items and the share")
    check(len(many(".expl")[0].text) > 40 if many(".expl") else False, "a plain-language explanation is shown")
    shot("5-split")

    click('[data-adjust="Aman"]'); set_value("#ov", "300"); css("#ov").send_keys(Keys.ENTER); settle(); time.sleep(0.4)
    check(bool(many(".stamp")), "an override shows the 'adjusted by hand' stamp")
    check(any("was" in e.get_attribute("class") for e in many(".was")), "the original amount is kept next to it")
    check("not covered" in d.find_element(By.ID, "main").text or "over-covered" in d.find_element(By.ID, "main").text, "the leftover is reported")
    shot("5-split-override")
    click("#rebalance"); settle(); time.sleep(0.4)
    finals = [money(e.text) for e in many(".person .head .amt")]
    check(abs(sum(finals) - big) < 0.005 and abs(finals[1] - 300) < 0.005, f"rebalancing spreads the difference: {finals}")
    click("#copy"); time.sleep(0.3)

    # mid-flow edits: remove a person who has items, then go back and change the bill (R3, R8)
    click(".steps button:nth-child(3)"); W.until(lambda x: text("h1") == "Who's eating")
    many(".people li .danger")[0].click(); time.sleep(0.2)
    check("?" in many(".people li .danger")[0].text, "removing someone with items asks first")
    many(".people li .danger")[0].click(); time.sleep(0.2)
    check(len(many(".people li")) == 2, "the person is removed after confirming")
    check(not d.find_element(By.CSS_SELECTOR, ".steps button:nth-child(5)").is_enabled(), "the split step locks until items are reassigned")

    errs = d.execute_script("return window.__errs")
    check(not errs, "no script errors: " + str(errs))
    # layout sanity: no horizontal scroll
    sw = d.execute_script("return [document.documentElement.scrollWidth, document.documentElement.clientWidth]")
    check(sw[0] <= sw[1] + 1, f"no horizontal scroll ({sw})")
finally:
    try: d.quit()
    except Exception: pass
print("\n%d check(s) failed" % len(fails) if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
