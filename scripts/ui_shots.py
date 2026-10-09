"""Screenshots of the five app steps with a real example photo through the real model, for the report. Needs the app running on 8765.
  python scripts/ui_shots.py"""
import os, time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
os.environ["TMPDIR"] = os.path.expanduser("~/ffprofile-tmp"); OUT = "report/figs"; PICK = 6          # bill_07: 7 items, CGST, SGST, rounding


def run(tag, size):
    o = Options(); o.add_argument("-headless"); o.binary_location = "/snap/firefox/current/usr/lib/firefox/firefox"
    o.set_preference("layout.css.prefers-color-scheme.content-override", 1)
    d = webdriver.Firefox(service=Service("/snap/bin/geckodriver"), options=o); d.set_window_size(*size)
    W = WebDriverWait(d, 60); many = lambda s: d.find_elements(By.CSS_SELECTOR, s)
    def shot(n, scroll=0):
        d.execute_script(f"window.scrollTo(0,{scroll})"); time.sleep(0.5); d.save_screenshot(f"{OUT}/ui_{tag}_{n}.png")
    try:
        d.get("http://localhost:8765/"); time.sleep(1.5); shot("1_photo")
        many(".examples .ex")[PICK].click(); W.until(lambda x: many(".item")); time.sleep(1.0)
        if tag == "phone": many(".peek summary")[0].click(); time.sleep(0.5)
        shot("2_check")
        if tag == "phone": shot("2_check_fields", 700)
        d.find_element(By.CSS_SELECTOR, "#to-people").click(); W.until(lambda x: d.find_element(By.CSS_SELECTOR, "h1").text == "Who's eating")
        for n in ("Riya", "Aman", "Sara"):
            e = d.find_element(By.CSS_SELECTOR, "#pname"); e.send_keys(n, Keys.ENTER); time.sleep(0.2)
        shot("3_people")
        d.find_element(By.CSS_SELECTOR, "#to-assign").click(); W.until(lambda x: d.find_element(By.CSS_SELECTOR, "h1").text == "Who had what")
        d.find_element(By.CSS_SELECTOR, "#equal-all").click(); time.sleep(0.3)
        for k in (0, 1):                                    # the first two dishes are Riya's alone
            for who in ("Aman", "Sara"):
                c = many(".assign-item")[k].find_element(By.CSS_SELECTOR, f'.chip[data-person="{who}"]'); d.execute_script("arguments[0].scrollIntoView({block:'center'})", c); c.click(); time.sleep(0.2)
        shot("4_assign")
        d.find_element(By.CSS_SELECTOR, "#to-split").click(); W.until(lambda x: many(".person")); time.sleep(0.8)
        shot("5_split", 0)
        mtx = d.find_element(By.CSS_SELECTOR, "#matrix"); y = d.execute_script("return arguments[0].getBoundingClientRect().top + window.scrollY", mtx)
        shot("5_matrix", int(y) - 90)
        print(tag, "errors:", d.execute_script("return window.__errs||[]"))
    finally:
        try: d.quit()
        except Exception: pass


run("phone", (390, 844)); run("wide", (1280, 900))
