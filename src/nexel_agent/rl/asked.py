"""What the words for a build ask of its shape, read so a design is judged on keeping it (live trial 4: the mind asked
for "a gabled roof made of shingles", the architect drew a flat one and it passed at full marks). Only what the words
name plainly is read - a roof's style, a round, a number of storeys -; what they leave out is the architect's call."""

import re

ROOF_WORDS = (("gable", r"\bgabled?\b|\bpitched\b|\bsloped?\b|\bslanted\b"),
              ("hip", r"\bhip(ped)?\b|\bdomed?\b"),
              ("flat", r"\bflat\b"))
ROUND = r"\bround\b|\bcircular\b|\bcircle\b"
WINGS = r"\b[ltu][- ]shaped?\b|\bcross[- ]shaped\b|\bwings?\b"
STOREYS = {"one": 1, "single": 1, "two": 2, "double": 2, "three": 3, "1": 1, "2": 2, "3": 3}


def asked(words: str) -> dict:
    """{"roof": style, "shape": "round"|"wings", "storeys": n}: what `words` name plainly, nothing else."""
    w, out = (words or "").lower(), {}
    roofs = [style for style, pat in ROOF_WORDS if re.search(pat + r"(?=[^.;,]*\broof)", w)]
    if len(roofs) == 1:                                        # (two named: "a flat or gabled roof", either will do)
        out["roof"] = roofs[0]
    if re.search(ROUND, w):
        out["shape"] = "round"
    elif re.search(WINGS, w):
        out["shape"] = "wings"
    if m := re.search(r"\b(one|single|two|double|three|[123])[- ]stor(?:e?y|ies|eys)\b", w):
        out["storeys"] = STOREYS[m.group(1)]
    return out


def unlike(spec: dict, want: dict) -> str:
    """How a design (a build of storeys) differs from what the words asked, "" if in nothing: "the words ask for a
    gable roof, this one is flat". A garden, a dug home or a compound are not read here."""
    if not want or any(spec.get(k) for k in ("parts", "garden", "underground")):
        return ""
    roof = spec.get("roof")
    style = None if not roof else "flat" if roof is True else roof if isinstance(roof, str) else roof.get("style", "flat")
    out = []
    if "roof" in want and style != want["roof"]:
        out.append(f"a {want['roof']} roof, this one is {style or 'open'}")
    if want.get("shape") == "round" and spec.get("shape", "rect") != "round":
        out.append("a round, this one is square")
    if want.get("shape") == "wings" and spec.get("shape", "rect") != "wings":
        out.append("wings (an L, a T, a cross), this one is one block")
    if "storeys" in want and int(spec.get("storeys", 1)) != want["storeys"]:
        out.append(f"{want['storeys']} storeys, this one has {int(spec.get('storeys', 1))}")
    return ("the words ask for " + "; for ".join(out)) if out else ""
