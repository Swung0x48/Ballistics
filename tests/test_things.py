"""The catalogue of things: every entry is well formed, every placed type of the game has one entry, and nothing local
(a path, a file name, an internal key of the local notes, a person) is in the repository."""
import json
import re

from ballistics import things
from ballistics.__main__ import main

# every type name the game's levels place (the placement tables of levels 1 to 13)
PLACED = ("P_Ball_Paper", "P_Ball_Stone", "P_Ball_Wood", "P_Box", "P_Dome", "P_Extra_Life", "P_Extra_Point", "P_Modul_01", "P_Modul_03",
          "P_Modul_08", "P_Modul_17", "P_Modul_18", "P_Modul_19", "P_Modul_25", "P_Modul_26", "P_Modul_29", "P_Modul_30", "P_Modul_34",
          "P_Modul_37", "P_Modul_41", "P_Trafo_Paper", "P_Trafo_Stone", "P_Trafo_Wood")


def test_every_entry_of_the_catalogue_is_well_formed():
    assert things.check() == {}


def test_every_placed_type_has_exactly_one_entry_and_no_entry_names_a_type_that_is_never_placed():
    cat = things.load()
    owners = {t: [th.id for th in cat.values() if t in th.types] for t in PLACED}
    assert all(len(v) == 1 for v in owners.values()), {t: v for t, v in owners.items() if len(v) != 1}
    for th in cat.values():
        assert set(th.types) <= set(PLACED), (th.id, th.types)
        for t in th.types:
            assert cat.of_type(t) is th


def test_the_three_balls_and_the_level_rules_are_there():
    cat = things.load()
    assert set(cat.balls()) == {"wood", "stone", "paper"}
    assert {"death_timing", "transform_timing", "sector_rebuild", "proximity_wake", "contact", "key_push"} <= set(cat.rules)
    for r in cat.rules.values():
        assert r["owner"] in cat, r["id"]


def test_every_state_machine_can_be_walked():
    for th in things.load().values():
        m = th.machine
        assert m["initial"] in th.states
        for t in m["transitions"]:
            assert t["from"] == "*" or t["from"] in th.states
            assert t["to"] == "*" or t["to"] in th.states
        assert th.after(m["initial"]) or len(th.states) == 1, th.id


def test_a_measured_fact_says_where_it_was_measured_and_no_fact_carries_a_path():
    cat = things.load()
    for where, d in list((th.id, {k: getattr(th, k) for k in things.TOP}) for th in cat.values()) + [("rules", cat.rules)]:
        for path, f in things.facts(d):
            assert "ref" not in f and "quote" not in f and "source" not in f, (where, path)
            if f["basis"] == "MEASURED":
                assert f.get("engine") in things.ENGINES, (where, path)
            if f["basis"] in ("READ", "MEASURED", "WIKI"):
                assert f.get("src") in things.SRC, (where, path)


def test_nothing_local_is_in_the_catalogue():
    bad = (
        (re.compile(r"[A-Za-z]:[/\\]|/Users/|geekerwan|swung|@[a-z0-9-]+\.(com|cn|org)|onlyforyou"), "a local path, an account, a person"),
        (re.compile(r"\b(data|out)/[\w./<>*\-]+"), "the path of a local file"),
        (re.compile(r"[\w<>*\-]+\.(txt|md|json|py|npz|nmo|cmo|raw|log|cxx|hxx|cpp|hpp|bmrc)\b"), "a file name"),
        (re.compile(r"canon|rulings"), "a key of the local notes"),
        (re.compile(r"\[\d+\]"), "an index into a local note"),
    )

    def strings(v):
        if isinstance(v, dict):
            for x in v.values():
                yield from strings(x)
        elif isinstance(v, list):
            for x in v:
                yield from strings(x)
        elif isinstance(v, str):
            yield v

    for f in sorted(things.DATA.glob("*.json")):
        for s in strings(json.load(open(f, encoding="utf8"))):
            for rx, what in bad:
                m = rx.search(s)
                assert m is None, "%s: %s: %r in %r" % (f.name, what, m.group(0), s[:80])


def test_the_command_lists_prints_and_checks(capsys):
    assert main(["things", "--check"]) == 0
    main(["things"])
    listing = capsys.readouterr().out
    assert "P_Modul_25" in listing and "ball_paper" in listing
    main(["things", "P_Trafo_Stone"])
    assert "变球" in capsys.readouterr().out
    main(["things", "death_timing"])
    assert "游戏秒" in capsys.readouterr().out
    assert main(["things", "no_such_thing"]) == 1
