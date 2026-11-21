"""The catalogue of everything in a level that is not plain road: the three player balls, the loose objects, the
mechanisms, the triggers, the pickups, and the rules that run a level. For each thing: its rigid bodies with their
numbers and collision hulls, the forces it applies, its state machine (what makes it change, what the change does,
how long it takes, what a death and a change of sector do to it), what each ball kind can do with it, how it changes
a route, and what is not known.

It is data, not simulation: nothing here moves. The kernel and the solver do not read it yet (docs/things.md).

Every number is a *fact*: {"v": value, "unit": "", "basis": ..., "src": ..., "engine": ...} with a basis that says
where it comes from:
    READ      read from the game itself (a script dump, a data table, a template file, or a note that records such a reading)
    MEASURED  measured in the engine (a calibration run, a recorded run)
    WIKI      only the community wiki says it
    DERIVED   follows from READ / MEASURED facts ("how_zh" says how)
    ASSUMED   not checked
    UNKNOWN   not known (v is None)
A READ, MEASURED or WIKI fact names the kind of source it rests on ("src", one of SRC below), and a MEASURED fact
also says where it was measured ("engine", one of ENGINES): the sources themselves are local game data and notes and
are not in the repository, so the catalogue names their kind and never a path. Frame counts say their frame rate.

    from ballistics import things
    cat = things.load()
    cat["P_Modul_25"].bodies[0]["mass"]["v"]
    cat.of_type("P_Trafo_Stone").machine["transitions"]
    cat.rules["death_timing"]
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).parent / "data"
BASES = ("READ", "MEASURED", "WIKI", "DERIVED", "ASSUMED", "UNKNOWN")
CATEGORIES = ("ball", "loose", "mechanism", "trigger", "pickup", "rule")
HULLS = ("sphere", "convex_hulls", "mesh", "box", "none")
MOTIONS = ("free", "hinge", "slider", "ball_joint", "fixed", "scripted", "spring", "none")
TRIGGERS = ("contact", "timer", "sector_enter", "sector_leave", "death", "script", "force", "proximity", "level_start",
            "level_end", "always")
EFFECTS = ("blocks", "opens", "carries", "launches", "requires_ball", "forbids_ball", "timing", "one_way", "hazard",
           "pickup", "changes_ball", "none")
FEATURES = ("rigid_body", "hinge", "slider", "ball_joint", "spring", "force_field", "trigger", "state_only",
            "shape_only", "none")
VERDICTS = ("passes", "pushes", "breaks", "rides", "blocked", "no_effect", "depends", "unknown")
UNKNOWN_KINDS = ("dump", "read", "measure", "owner", "host2002", "tool")
BALLS = ("wood", "stone", "paper")
# the kind of source a READ / MEASURED / WIKI fact rests on
SRC = ("table",        # a data table of the game (placement rows, level arrays, the ball table)
       "dump",         # a dump of a script of the game: its blocks, parameters and links
       "template",     # a template file of the game (the placeholder objects and their frames)
       "source",       # the source code of the rebuilt engine and of the physics library
       "wiki",         # the community wiki
       "record",       # a recorded run (states per frame) or the state specification read from one
       "measurement",  # a run made to measure this, reduced in a local note
       "answer",       # an answer from the people who run the other tools (retail measurements)
       "owner",        # the project owner's statement
       "map",          # the coarse layer's map of a level
       "note",         # another local note
       "docs", "code")  # this repository
# where a MEASURED fact was measured
ENGINES = ("original",   # records of the 2002 physics library replaying tool-assisted runs
           "retail",     # the retail game
           "x87_soft",   # the rebuilt engine in x87 arithmetic, replay mode (bit-identical to "original" in agent mode only)
           "rebuilt")    # the rebuilt engine in portable arithmetic, 66 Hz tool frames: same rules, different rounding
BODY_FACTS = ("mass", "friction", "elasticity", "lin_damp", "rot_damp", "group", "starts")
TOP = ("id", "name_zh", "category", "types", "summary_zh", "uses", "bodies", "forces", "numbers", "machine",
       "by_ball", "route_effects", "model", "unknowns")


def is_fact(d) -> bool:
    return isinstance(d, dict) and "basis" in d


def facts(v, path=""):
    """Every fact under v, as (path, fact)."""
    if isinstance(v, dict):
        if "basis" in v:
            yield path, v
        for k, x in v.items():
            yield from facts(x, "%s/%s" % (path, k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from facts(x, "%s/%d" % (path, i))


def _check_fact(d: dict, where: str, out: list):
    b = d.get("basis")
    if b not in BASES:
        out.append("%s: basis %r is not one of %s" % (where, b, ", ".join(BASES)))
        return
    if "quote" in d or "source" in d or "ref" in d:
        out.append("%s: a catalogue fact names the kind of its source (src), never a path or the text of a file" % where)
    if b in ("READ", "MEASURED", "WIKI"):
        if d.get("src") not in SRC:
            out.append("%s: basis %s needs src (one of %s)" % (where, b, ", ".join(SRC)))
        elif b == "WIKI" and d["src"] != "wiki":
            out.append("%s: basis WIKI but src is %r" % (where, d["src"]))
        if b == "MEASURED" and d.get("engine") not in ENGINES:
            out.append("%s: basis MEASURED needs engine (one of %s)" % (where, ", ".join(ENGINES)))
    elif b == "DERIVED" and not d.get("how_zh"):
        out.append("%s: basis DERIVED needs how_zh" % where)
    elif b == "UNKNOWN" and d.get("v") is not None:
        out.append("%s: basis UNKNOWN but a value is given" % where)


def check_machine(sm, where: str, out: list):
    if not isinstance(sm, dict):
        out.append("%s must be an object" % where)
        return
    names = [s.get("name") for s in sm.get("states") or [] if isinstance(s, dict)]
    if not names:
        out.append("%s.states is empty (a thing that never changes has one state)" % where)
    if len(set(names)) != len(names):
        out.append("%s.states: a state name is used twice" % where)
    for i, f in enumerate(sm.get("flags") or []):
        if not isinstance(f, dict) or not f.get("name") or not f.get("meaning_zh"):
            out.append("%s.flags[%d] needs name and meaning_zh (a flag is a boolean that holds across states)" % (where, i))
    if sm.get("initial") not in names:
        out.append("%s.initial %r is not a state" % (where, sm.get("initial")))
    reached = {sm.get("initial")}
    for i, t in enumerate(sm.get("transitions") or []):
        w = "%s.transitions[%d]" % (where, i)
        for end in ("from", "to"):
            if t.get(end) != "*" and t.get(end) not in names:
                out.append("%s.%s %r is not a state (or \"*\")" % (w, end, t.get(end)))
        if t.get("trigger") not in TRIGGERS:
            out.append("%s.trigger %r is not one of %s" % (w, t.get("trigger"), ", ".join(TRIGGERS)))
        if not t.get("condition_zh") or not t.get("effect_zh"):
            out.append("%s needs condition_zh and effect_zh" % w)
        if "seconds" in t and t["seconds"] is not None and not isinstance(t["seconds"], (int, float)):
            out.append("%s.seconds must be a number of game seconds or null" % w)
        if "basis" not in t:
            out.append("%s must carry a basis like a fact" % w)
        reached.add(t.get("to"))
    for n in names:
        if n not in reached and "*" not in reached:
            out.append("%s: no transition leads to state %r" % (where, n))
    r = sm.get("reset")
    if not isinstance(r, dict) or not r.get("on_death_zh") or not r.get("on_sector_change_zh") or "basis" not in r:
        out.append("%s.reset needs on_death_zh, on_sector_change_zh and a basis" % where)


def check_entry(d, rules=(), name: str | None = None) -> list:
    """The problems of one catalogue entry (an empty list if it is well formed)."""
    out = []
    if not isinstance(d, dict):
        return ["the file must hold one JSON object"]
    for k in TOP:
        if k not in d:
            out.append("missing key %r" % k)
    if name is not None and d.get("id") != name:
        out.append("the file is named %s but its id is %r" % (name, d.get("id")))
    if d.get("category") not in CATEGORIES:
        out.append("category %r is not one of %s" % (d.get("category"), ", ".join(CATEGORIES)))
    if not isinstance(d.get("types"), list):
        out.append("types must be a list (the placed type names this entry covers; empty for a ball or a rule)")
    for u in d.get("uses") or []:
        if rules and u not in rules:
            out.append("uses: no rule %r" % u)
    for i, b in enumerate(d.get("bodies") or []):
        w = "bodies[%d]" % i
        if not isinstance(b, dict) or not b.get("name") or not b.get("role_zh"):
            out.append("%s needs name and role_zh" % w)
            continue
        h = b.get("hull") or {}
        if h.get("kind") not in HULLS:
            out.append("%s.hull.kind %r is not one of %s" % (w, h.get("kind"), ", ".join(HULLS)))
        if "basis" not in h:
            out.append("%s.hull must carry a basis" % w)
        if "vertices" in h or "faces" in h:
            out.append("%s.hull: no geometry in the catalogue, only mesh names and sizes" % w)
        mo = b.get("motion") or {}
        if mo.get("kind") not in MOTIONS:
            out.append("%s.motion.kind %r is not one of %s" % (w, mo.get("kind"), ", ".join(MOTIONS)))
        if "basis" not in mo:
            out.append("%s.motion must carry a basis" % w)
        for k in BODY_FACTS:
            if not is_fact(b.get(k)):
                out.append("%s.%s must be a fact (basis UNKNOWN and v null if not known)" % (w, k))
    for i, f in enumerate(d.get("forces") or []):
        if not isinstance(f, dict) or not f.get("name") or not f.get("law_zh") or "basis" not in f:
            out.append("forces[%d] needs name, law_zh and a basis" % i)
    nums = d.get("numbers")
    if not isinstance(nums, dict):
        out.append("numbers must be an object of named facts (may be empty)")
    else:
        for k, f in nums.items():
            if not is_fact(f) or not f.get("label_zh"):
                out.append("numbers.%s must be a fact with label_zh" % k)
    check_machine(d.get("machine"), "machine", out)
    bb = d.get("by_ball")
    if not isinstance(bb, dict) or any(k not in bb for k in BALLS):
        out.append("by_ball needs the keys wood, stone and paper")
    else:
        for k in BALLS:
            if bb[k].get("verdict") not in VERDICTS or not bb[k].get("can_zh") or "basis" not in bb[k]:
                out.append("by_ball.%s needs verdict (%s), can_zh and a basis" % (k, ", ".join(VERDICTS)))
    for i, r in enumerate(d.get("route_effects") or []):
        if r.get("kind") not in EFFECTS or not r.get("effect_zh") or "basis" not in r:
            out.append("route_effects[%d] needs kind (%s), effect_zh and a basis" % (i, ", ".join(EFFECTS)))
    for i, r in enumerate(d.get("model") or []):
        if r.get("feature") not in FEATURES or not r.get("need_zh") or not r.get("minimal_zh"):
            out.append("model[%d] needs feature (%s), need_zh and minimal_zh" % (i, ", ".join(FEATURES)))
    for i, r in enumerate(d.get("unknowns") or []):
        if r.get("kind") not in UNKNOWN_KINDS or not r.get("question_zh") or not r.get("matters_zh") or not r.get("how_zh"):
            out.append("unknowns[%d] needs kind (%s), question_zh, matters_zh and how_zh" % (i, ", ".join(UNKNOWN_KINDS)))
    for path, f in facts(d):
        _check_fact(f, path, out)
    return out


def check_rules(d) -> list:
    out = []
    if not isinstance(d, dict) or not isinstance(d.get("rules"), list):
        return ["rules.json must hold {\"rules\": [...]}"]
    seen = set()
    for i, r in enumerate(d["rules"]):
        w = "rules[%d]" % i
        if not r.get("id") or not r.get("name_zh") or not r.get("statement_zh") or not r.get("owner"):
            out.append("%s needs id, name_zh, statement_zh and owner" % w)
        if r.get("id") in seen:
            out.append("%s: id %r is used twice" % (w, r.get("id")))
        seen.add(r.get("id"))
        if not isinstance(r.get("numbers", {}), dict):
            out.append("%s.numbers must be an object of facts" % w)
    for path, f in facts(d):
        _check_fact(f, path, out)
    return out


@dataclass
class Thing:
    id: str
    name_zh: str
    category: str
    types: list
    summary_zh: str
    uses: list
    bodies: list
    forces: list
    numbers: dict
    machine: dict
    by_ball: dict
    route_effects: list
    model: list
    unknowns: list
    extra: dict = field(default_factory=dict)

    @property
    def states(self) -> list:
        return [s["name"] for s in self.machine["states"]]

    def after(self, state: str) -> list:
        """The transitions that can be taken from a state."""
        return [t for t in self.machine["transitions"] if t["from"] in (state, "*")]

    def facts(self):
        return facts({k: getattr(self, k) for k in ("bodies", "forces", "numbers", "machine", "by_ball", "route_effects")})

    def count(self) -> dict:
        """How many facts rest on each basis."""
        out = dict.fromkeys(BASES, 0)
        for _, f in self.facts():
            out[f["basis"]] += 1
        return out


class Catalogue(dict):
    """id -> Thing; .rules: id -> the shared rules, each stored once."""

    def __init__(self, things, rules):
        super().__init__(things)
        self.rules = rules
        self._types = {t: th for th in things.values() for t in th.types}

    def of_type(self, type_name: str):
        """The entry of a placed type name ("P_Trafo_Stone", "P_Modul_25", "P_Box"), or None."""
        return self._types.get(type_name)

    def balls(self) -> dict:
        return {th.id.split("_", 1)[1]: th for th in self.values() if th.category == "ball"}


def load(folder: Path | None = None) -> Catalogue:
    folder = Path(folder) if folder else DATA
    rules = {r["id"]: r for r in json.load(open(folder / "rules.json", encoding="utf8"))["rules"]}
    things = {}
    for f in sorted(folder.glob("*.json")):
        if f.name == "rules.json":
            continue
        d = json.load(open(f, encoding="utf8"))
        things[d["id"]] = Thing(**{k: d[k] for k in TOP}, extra={k: v for k, v in d.items() if k not in TOP})
    return Catalogue(things, rules)


def check(folder: Path | None = None) -> dict:
    """{file name: problems} for every file of the catalogue that has any."""
    folder = Path(folder) if folder else DATA
    bad = {}
    rules = json.load(open(folder / "rules.json", encoding="utf8"))
    p = check_rules(rules)
    if p:
        bad["rules.json"] = p
    ids = {r.get("id") for r in rules.get("rules", [])}
    for f in sorted(folder.glob("*.json")):
        if f.name == "rules.json":
            continue
        try:
            d = json.load(open(f, encoding="utf8"))
        except Exception as e:
            bad[f.name] = ["not valid JSON: %s" % e]
            continue
        p = check_entry(d, ids, f.stem)
        if p:
            bad[f.name] = p
    return bad


_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_MARK = {"READ": "读", "MEASURED": "测", "WIKI": "wiki", "DERIVED": "推", "ASSUMED": "假定", "UNKNOWN": "未知"}
_ENGINE = {"original": "原版记录", "retail": "零售版", "x87_soft": "x87 回放", "rebuilt": "重建引擎"}


def _mark(f) -> str:
    """The basis of a fact and, for a measured one, where it was measured: 【测·原版记录】."""
    return "【%s%s】" % (_MARK[f["basis"]], ("·" + _ENGINE[f["engine"]]) if f.get("engine") in _ENGINE else "")


def _val(f) -> str:
    v = f.get("v")
    s = "不知道" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))
    return "%s%s%s" % (s, (" " + f["unit"]) if f.get("unit") else "", _mark(f))


def render(th: Thing, rules: dict | None = None) -> str:
    """One entry as Markdown, in Chinese."""
    L = ["# %s（%s）" % (th.name_zh, th.id), "", th.summary_zh, ""]
    if th.types:
        L += ["摆放类型：" + "、".join(th.types), ""]
    if th.uses:
        L += ["共用规则：" + "、".join("%s（%s）" % (rules[u]["name_zh"], u) if rules and u in rules else u for u in th.uses), ""]
    if th.bodies:
        L += ["## 刚体", "", "| 刚体 | 作用 | 形状 | 质量 | 摩擦 | 弹性 | 线性阻尼 | 旋转阻尼 | 碰撞组 | 初始 | 运动 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in th.bodies:
            h = b["hull"]
            shape = "%s%s%s%s" % (h["kind"], (" " + "、".join(h["meshes"])) if h.get("meshes") else "",
                                  (" 尺寸 %s" % h["size"]) if h.get("size") else "", _mark(h))
            mo = b["motion"]
            L.append("| %s | %s | %s | %s | %s%s |" % (b["name"], b["role_zh"], shape, " | ".join(_val(b[k]) for k in BODY_FACTS),
                                                    mo["kind"], _mark(mo)))
        L.append("")
        for b in th.bodies:
            for k in ("hull", "motion"):
                n = b[k].get("note_zh")
                if n:
                    L.append("- %s 的%s：%s" % (b["name"], "形状" if k == "hull" else "运动", n))
        L.append("")
    if th.forces:
        L += ["## 力", ""]
        for f in th.forces:
            L.append("- **%s**%s：%s%s" % (f["name"], _mark(f), f["law_zh"], ("作用范围：" + f["region_zh"]) if f.get("region_zh") else ""))
            for k, v in (f.get("values") or {}).items():
                if is_fact(v):
                    L.append("  - %s：%s%s" % (v.get("label_zh", k), _val(v), ("。" + v["note_zh"]) if v.get("note_zh") else ""))
        L.append("")
    if th.numbers:
        L += ["## 数值", "", "| 量 | 值 | 说明 |", "|---|---|---|"]
        for k, f in th.numbers.items():
            L.append("| %s | %s | %s |" % (f["label_zh"], _val(f), (f.get("note_zh") or f.get("how_zh") or "").replace("\n", " ")))
        L.append("")
    sm = th.machine
    L += ["## 状态机", "", "初始状态：%s" % sm["initial"], ""]
    for s in sm["states"]:
        L.append("- **%s**：%s" % (s["name"], s["meaning_zh"]))
    for f in sm.get("flags") or []:
        L.append("- 标志 **%s**：%s" % (f["name"], f["meaning_zh"]))
    if sm.get("transitions"):
        L += ["", "| 从 | 到 | 触发 | 条件 | 结果 | 用时（游戏秒） | 依据 |", "|---|---|---|---|---|---|---|"]
        for t in sm["transitions"]:
            sec = t.get("seconds")
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (t["from"], t["to"], t["trigger"], t["condition_zh"], t["effect_zh"],
                                                             ("%g" % sec) if isinstance(sec, (int, float)) else (t.get("time_zh") or "—"), _mark(t)[1:-1]))
    r = sm["reset"]
    L += ["", "- 死亡后%s：%s" % (_mark(r), r["on_death_zh"]), "- 换小节%s：%s" % (_mark(r), r["on_sector_change_zh"]), ""]
    L += ["## 三种球", ""]
    for k, zh in zip(BALLS, ("木球", "石球", "纸球")):
        b = th.by_ball[k]
        L.append("- **%s**（%s）%s：%s" % (zh, b["verdict"], _mark(b), b["can_zh"]))
    if th.route_effects:
        L += ["", "## 对路线的影响", ""]
        L += ["- （%s）%s%s" % (r["kind"], _mark(r), r["effect_zh"]) for r in th.route_effects]
    if th.model:
        L += ["", "## 要模拟它需要什么", ""]
        L += ["- （%s）%s 最小模型：%s" % (m["feature"], m["need_zh"], m["minimal_zh"]) for m in th.model]
    if th.unknowns:
        L += ["", "## 不知道的", ""]
        L += ["- （%s）%s 影响：%s 怎样查清：%s" % (u["kind"], u["question_zh"], u["matters_zh"], u["how_zh"]) for u in th.unknowns]
    return "\n".join(L) + "\n"


def render_rule(r: dict) -> str:
    """One shared rule as Markdown, in Chinese."""
    L = ["# %s（%s）" % (r["name_zh"], r["id"]), "", r["statement_zh"], "", "归属：%s" % r["owner"], ""]
    if r.get("numbers"):
        L += ["| 量 | 值 | 说明 |", "|---|---|---|"]
        for k, f in r["numbers"].items():
            L.append("| %s | %s | %s |" % (f["label_zh"], _val(f), (f.get("note_zh") or f.get("how_zh") or "").replace(chr(10), " ")))
    return chr(10).join(L) + chr(10)
