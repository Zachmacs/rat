"""Tests for the Capital.com price source and the short side of entry.py.

    python3 -m unittest discover -s .claude/skills/trader-entry/tests -v

Offline only: Capital.com's HTTP calls are replaced by a fake, and the short
side is checked on real Yahoo ^DJI 5-minute bars for Wed 2026-11-11.
"""

import contextlib
import datetime
import importlib.util
import io
import json
import os
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
DJI = os.path.join(FIX, "DJI-2026-11-11-5m.json.gz")

_spec = importlib.util.spec_from_file_location("trader_entry", os.path.join(os.path.dirname(HERE), "entry.py"))
entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry)


def utc(text):
    return entry.parse_now(text)


def cap_bar(et, o, h, l, c, spread=2.0, v=100):
    """One Capital.com price row for a New York time, bid/ask around a mid."""
    t = entry.eastern_to_utc(et)
    half = spread / 2.0

    def p(x):
        return {"bid": x - half, "ask": x + half}

    return {"snapshotTime": t.strftime("%Y-%m-%dT%H:%M:%S"), "snapshotTimeUTC": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "openPrice": p(o), "highPrice": p(h), "lowPrice": p(l), "closePrice": p(c), "lastTradedVolume": v}


def cap_day(day, closes, step=5):
    """Bars from 9:30, one per close: the range bars sit between 100 and 110."""
    start = datetime.datetime.combine(day, datetime.time(9, 30))
    rows = []
    for i, c in enumerate(closes):
        et = start + datetime.timedelta(minutes=step * i)
        rows.append(cap_bar(et, c, max(c, 110.0) if i < 3 else c + 0.5, min(c, 100.0) if i < 3 else c - 0.5, c))
    return {"prices": rows}


DAY = datetime.date(2026, 11, 12)   # a Thursday, a normal NYSE day


class ParseCapital(unittest.TestCase):
    def test_mid_prices_and_order(self):
        doc = cap_day(DAY, [105, 104, 106, 108])
        doc["prices"].reverse()
        ch = entry.parse_capital(json.dumps(doc), "US30", "5m")
        self.assertEqual(ch["source"], "capital")
        self.assertEqual([b["c"] for b in ch["bars"]], [105, 104, 106, 108])   # sorted, mid of bid/ask
        first = entry.to_eastern(entry.utc_of(ch["bars"][0]["t"]))
        self.assertEqual(first, datetime.datetime(2026, 11, 12, 9, 30))
        self.assertEqual(ch["bars"][0]["h"], 110.0)
        self.assertEqual(ch["bars"][0]["l"], 100.0)
        self.assertEqual(ch["data_time"], ch["bars"][-1]["t"])   # a saved file: the last bar may be forming

    def test_one_sided_and_bad_rows(self):
        doc = {"prices": [{"snapshotTimeUTC": "2026-11-12T14:30:00", "closePrice": {"bid": 50.0}},
                          {"snapshotTimeUTC": "bad", "closePrice": {"bid": 1, "ask": 2}},
                          {"snapshotTimeUTC": "2026-11-12T14:35:00", "closePrice": {}}]}
        ch = entry.parse_capital(json.dumps(doc), "SILVER", "5m")
        self.assertEqual([b["c"] for b in ch["bars"]], [50.0])

    def test_not_capital(self):
        with self.assertRaises(entry.EntryError):
            entry.parse_capital("{}", "US30", "5m")

    def test_prices_path_is_the_session_in_utc(self):
        path = entry.capital_prices_path("US30", "5m", DAY)
        self.assertEqual(path, "/prices/US30?resolution=MINUTE_5&max=1000"
                               "&from=2026-11-12T14:30:00&to=2026-11-12T21:00:00")
        self.assertIn("resolution=MINUTE&", entry.capital_prices_path("US30", "1m", DAY))


class Sides(unittest.TestCase):
    def chart(self, closes):
        return entry.parse_capital(json.dumps(cap_day(DAY, closes)), "US30", "5m")

    def test_long_triggers_above_the_high(self):
        res = entry.evaluate(self.chart([105, 104, 106, 108, 110, 111, 99]), utc("2026-11-12 10:30"))
        self.assertEqual(res["status"], "triggered")
        self.assertEqual(res["level"], 110.0)
        self.assertEqual(res["trigger"]["price"], 111.0)        # 110 is AT the high, not above it
        self.assertEqual(res["trigger"]["time"], "10:00 AM")

    def test_short_triggers_below_the_low(self):
        res = entry.evaluate(self.chart([105, 104, 106, 103, 100, 99, 111]), utc("2026-11-12 10:30"), side="sell")
        self.assertEqual(res["side"], "sell")
        self.assertEqual(res["level"], 100.0)
        self.assertEqual(res["trigger"]["price"], 99.0)          # 100 is AT the low, not below it
        self.assertEqual(res["trigger"]["time"], "10:00 AM")
        self.assertLess(res["trigger"]["pct_over"], 0)
        lines = entry.describe(res)
        self.assertIn("breakdown, short side", lines[0])
        self.assertIn("Capital.com", lines[0])
        self.assertTrue(any("closed below the range low $100.00" in x for x in lines))
        self.assertTrue(any("stop ABOVE and the target BELOW" in x for x in lines))

    def test_short_waiting_says_how_far(self):
        res = entry.evaluate(self.chart([105, 104, 106, 103, 102]), utc("2026-11-12 09:57"), side="sell")
        self.assertEqual(res["status"], "waiting")
        self.assertIn("$2.00 over the range low $100.00", res["reason"])

    def test_short_level(self):
        res = entry.evaluate(self.chart([105, 104, 106, 103, 101.5]), utc("2026-11-12 10:00"),
                             setup="level", level=102, side="sell")
        self.assertEqual(res["trigger"]["price"], 101.5)
        self.assertEqual(entry.setup_name(res), "close below $102.00")

    def test_short_expires(self):
        res = entry.evaluate(self.chart([105] * 78), utc("2026-11-12 15:45"), side="sell")
        self.assertEqual(res["status"], "expired")
        self.assertIn("no close below the range low $100.00", res["reason"])

    def test_forming_capital_bar_does_not_count(self):
        ch = self.chart([105, 104, 106, 103, 99])
        ch["data_time"] = entry.ts_of(utc("2026-11-12 09:52"))   # live: fetched mid-bar
        res = entry.evaluate(ch, utc("2026-11-12 09:52"), side="sell")
        self.assertEqual(res["status"], "waiting")
        self.assertEqual(res["forming"]["price"], 99.0)

    def test_real_dow_short(self):
        """Wed 2026-11-11 ^DJI: the 10:05 bar closed 52,131.13, under the range low 52,132.55."""
        ch = entry.get_chart("^DJI", "5m", offline=DJI)
        res = entry.evaluate(ch, utc("2026-11-11 15:30"), side="sell")
        self.assertEqual(res["opening_range"]["low"], 52132.55)
        self.assertEqual(res["trigger"]["time"], "10:10 AM")
        self.assertEqual(res["trigger"]["price"], 52131.13)
        long_ = entry.evaluate(ch, utc("2026-11-11 15:30"))
        self.assertEqual(long_["trigger"]["time"], "12:05 PM")
        self.assertEqual(long_["trigger"]["price"], 52352.35)


class FakeCapital(object):
    """Stands in for _capital_http: records calls, answers like Capital.com."""

    def __init__(self, prices, expire_once=False):
        self.calls, self.prices, self.expire_once, self.logins = [], prices, expire_once, 0

    def __call__(self, method, url, headers, body=None, timeout=20):
        self.calls.append((method, url, dict(headers), body))
        if url.endswith("/session") and method == "POST":
            if headers.get("X-CAP-API-KEY") != "k" or body != {"identifier": "me@example.com", "password": "pw",
                                                                "encryptedPassword": False}:
                return 401, {}, '{"errorCode":"error.invalid.details"}'
            self.logins += 1
            return 200, {"CST": "cst%d" % self.logins, "X-SECURITY-TOKEN": "sec%d" % self.logins}, "{}"
        if headers.get("CST") != "cst%d" % self.logins:
            return 401, {}, '{"errorCode":"error.invalid.session.token"}'
        if self.expire_once:
            self.expire_once = False
            return 401, {}, '{"errorCode":"error.invalid.session.token"}'
        if "/prices/NOPE" in url:
            return 404, {}, '{"errorCode":"error.not-found.epic"}'
        if "/markets?" in url:
            return 200, {}, json.dumps({"markets": [{"epic": "US30", "instrumentName": "US Wall Street 30",
                                                     "instrumentType": "INDICES", "marketStatus": "TRADEABLE"}]})
        return 200, {}, json.dumps(self.prices)


ENV = {"CAPITAL_API_KEY": "k", "CAPITAL_IDENTIFIER": "me@example.com", "CAPITAL_PASSWORD": "pw"}


class CapitalHttp(unittest.TestCase):
    def setUp(self):
        entry._CAPITAL.clear()

    def run_env(self, env, fake, fn):
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(entry, "_capital_http", fake):
            return fn()

    def test_login_then_prices_on_demo(self):
        fake = FakeCapital(cap_day(DAY, [105, 104, 106, 111]))
        ch = self.run_env(ENV, fake, lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertEqual(len(ch["bars"]), 4)
        self.assertTrue(ch["demo"])
        self.assertTrue(fake.calls[0][1].startswith("https://demo-api-capital.backend-capital.com/api/v1/session"))
        self.assertEqual(fake.calls[1][2]["X-SECURITY-TOKEN"], "sec1")
        self.assertNotIn("X-CAP-API-KEY", fake.calls[1][2])

    def test_live_only_when_asked(self):
        fake = FakeCapital(cap_day(DAY, [105]))
        env = dict(ENV, CAPITAL_DEMO="0")
        ch = self.run_env(env, fake, lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertFalse(ch["demo"])
        self.assertTrue(fake.calls[0][1].startswith("https://api-capital.backend-capital.com/"))

    def test_expired_session_logs_in_again(self):
        fake = FakeCapital(cap_day(DAY, [105]), expire_once=True)
        self.run_env(ENV, fake, lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertEqual(fake.logins, 2)

    def test_missing_keys_say_which(self):
        with self.assertRaises(entry.EntryError) as e:
            self.run_env({"CAPITAL_API_KEY": "k"}, FakeCapital({}),
                         lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertIn("CAPITAL_IDENTIFIER, CAPITAL_PASSWORD", str(e.exception))

    def test_bad_login_and_unknown_epic(self):
        with self.assertRaises(entry.EntryError) as e:
            self.run_env(dict(ENV, CAPITAL_PASSWORD="wrong"), FakeCapital({}),
                         lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertIn("error.invalid.details", str(e.exception))
        self.assertIn("DEMO", str(e.exception))
        entry._CAPITAL.clear()
        with self.assertRaises(entry.EntryError) as e:
            self.run_env(ENV, FakeCapital({}),
                         lambda: entry.get_chart("NOPE", "5m", DAY, source="capital", today=DAY))
        self.assertIn("epics", str(e.exception))

    def test_credentials_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "capital.json")
            with open(f, "w") as fh:
                json.dump({"CAPITAL_API_KEY": "k", "CAPITAL_IDENTIFIER": "me@example.com", "CAPITAL_PASSWORD": "pw"}, fh)
            fake = FakeCapital(cap_day(DAY, [105]))
            ch = self.run_env({"CAPITAL_CREDENTIALS": f}, fake,
                              lambda: entry.get_chart("US30", "5m", DAY, source="capital", today=DAY))
        self.assertEqual(len(ch["bars"]), 1)
        self.assertEqual(fake.logins, 1)

    def test_environment_beats_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "capital.json")
            with open(f, "w") as fh:
                json.dump({"CAPITAL_API_KEY": "file", "CAPITAL_IDENTIFIER": "x", "CAPITAL_PASSWORD": "y"}, fh)
            with mock.patch.dict(os.environ, dict(ENV, CAPITAL_CREDENTIALS=f), clear=True):
                self.assertEqual(entry.capital_credentials()["CAPITAL_API_KEY"], "k")

    def test_missing_file_says_where(self):
        with mock.patch.dict(os.environ, {"CAPITAL_CREDENTIALS": "/nonexistent/capital.json"}, clear=True):
            with self.assertRaises(entry.EntryError) as e:
                entry.capital_login()
        self.assertIn("/nonexistent/capital.json", str(e.exception))

    def test_epics(self):
        rows = self.run_env(ENV, FakeCapital({}), lambda: entry.capital_epics("wall street"))
        self.assertEqual(rows[0][0], "US30")


class Cli(unittest.TestCase):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = entry.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_short_check_writes_a_sell_entry(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "entry.json")
            code, out, _ = self.run_cli(["check", "^DJI", "--side", "short", "--offline", DJI,
                                         "--now", "2026-11-11 10:12", "--write", path,
                                         "--config", os.path.join(d, "none.json")])
            self.assertEqual(code, 0)
            self.assertIn("TRIGGERED 10:10 AM at $52131.13", out)
            doc = json.load(open(path))
        self.assertEqual(doc["side"], "sell")
        self.assertEqual(doc["price"], 52131.13)
        self.assertEqual(doc["source"], "yahoo")

    def test_offline_capital_file_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "US30.json")
            with open(f, "w") as fh:
                json.dump(cap_day(DAY, [105, 104, 106, 103, 99]), fh)
            code, out, _ = self.run_cli(["check", "US30", "--side", "sell", "--offline", d,
                                         "--now", "2026-11-12 10:00", "--config", os.path.join(d, "none.json")])
        self.assertEqual(code, 0)
        self.assertIn("Capital.com", out)
        self.assertIn("closed below the range low", out)

    def test_short_watch_replays(self):
        code, out, _ = self.run_cli(["watch", "^DJI", "--side", "sell", "--offline", DJI,
                                     "--now", "2026-11-11 09:31", "--out", os.devnull])
        self.assertEqual(code, 0)
        self.assertIn("TRIGGERED: closed below $52132.55", out)

    def test_epics_needs_capital(self):
        code, _, err = self.run_cli(["epics", "silver", "--source", "yahoo"])
        self.assertEqual(code, 1)
        self.assertIn("--source capital", err)


if __name__ == "__main__":
    unittest.main()
