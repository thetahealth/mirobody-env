"""EDF / EDF+ for the rare-coding EEG attachments: read the header, write a cut as plain EDF or
EDF+C, read the annotations back.

What a writer may change, and nothing else:
* the patient and recording identification fields (80 characters each) -- de-identification
  and the EDF+ subfield syntax;
* for EDF+C: the reserved field ("EDF+C"), the header length and the signal count, because one
  "EDF Annotations" signal is appended. Every original signal header field and every original
  data sample stays byte for byte; each data record grows by the annotation signal's bytes at
  its end. The first TAL of every record is the timekeeping TAL (record onset, EDF+ spec 2.2.4);
  each annotation sits in the record that contains its onset.

Onsets are seconds relative to the file start. No signal data is decoded here.
SYNTHETIC evaluation data only.
"""
from __future__ import annotations

import math
import pathlib
from collections import Counter
from dataclasses import dataclass, field

ANNOT_LABEL = "EDF Annotations"
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
_SIGNAL_FIELDS = (("label", 16), ("transducer", 80), ("physical_dimension", 8), ("physical_min", 8),
                  ("physical_max", 8), ("digital_min", 8), ("digital_max", 8), ("prefilter", 80),
                  ("samples", 8), ("reserved", 32))


@dataclass
class Header:
    version: str
    patient: str
    recording: str
    startdate: str
    starttime: str
    header_bytes: int
    reserved: str
    n_records: int
    record_duration: float
    ns: int
    signals: list[dict] = field(default_factory=list)      # raw field strings, `samples` as int
    raw_signal_fields: list[list[bytes]] = field(default_factory=list)   # per field, per signal

    @property
    def edfplus(self) -> bool:
        return self.reserved.startswith("EDF+")

    @property
    def record_bytes(self) -> int:
        return sum(2 * s["samples"] for s in self.signals)

    def data_signals(self) -> list[dict]:
        return [s for s in self.signals if s["label"] != ANNOT_LABEL]


def _num(b: bytes) -> float:
    return float(b.decode("ascii").strip())


def read_header(path: str | pathlib.Path) -> Header:
    with open(path, "rb") as fh:
        main = fh.read(256)
        if len(main) < 256:
            raise ValueError(f"{path}: shorter than an EDF main header")
        ns = int(_num(main[252:256]))
        rest = fh.read(256 * ns)
    h = Header(version=main[0:8].decode("ascii").strip(), patient=main[8:88].decode("ascii", "replace").strip(),
               recording=main[88:168].decode("ascii", "replace").strip(),
               startdate=main[168:176].decode("ascii").strip(), starttime=main[176:184].decode("ascii").strip(),
               header_bytes=int(_num(main[184:192])), reserved=main[192:236].decode("ascii", "replace").strip(),
               n_records=int(_num(main[236:244])), record_duration=_num(main[244:252]), ns=ns)
    if h.header_bytes != 256 * (ns + 1):
        raise ValueError(f"{path}: header length {h.header_bytes} != 256*(ns+1)")
    pos = 0
    cols: list[list[bytes]] = []
    for _, width in _SIGNAL_FIELDS:
        cols.append([rest[pos + i * width: pos + (i + 1) * width] for i in range(ns)])
        pos += width * ns
    h.raw_signal_fields = cols
    for i in range(ns):
        s = {name: cols[k][i].decode("ascii", "replace").strip() for k, (name, _) in enumerate(_SIGNAL_FIELDS)}
        s["samples"] = int(float(s["samples"]))
        h.signals.append(s)
    return h


# ---------------------------------------------------------------- EDF+ identification fields
def edfplus_date(startdate: str) -> str:
    """Header `dd.mm.yy` -> EDF+ `DD-MMM-YYYY` (EDF+ clipping: yy 85-99 = 19yy, else 20yy)."""
    d, m, y = startdate.split(".")
    yy = int(y)
    return f"{int(d):02d}-{_MONTHS[int(m) - 1]}-{(1900 if yy >= 85 else 2000) + yy}"


def _subfield(v: str) -> str:
    v = str(v).strip().replace(" ", "_")
    return v or "X"


def edfplus_patient(p: dict | None) -> str:
    """EDF+ patient field `code sex birthdate name`; None -> all unknown (`X X X X`). A code, sex
    or birthdate containing a space is refused (the subfields are space separated)."""
    if not p:
        return "X X X X"
    for k in ("code", "sex", "birthdate"):
        if " " in str(p.get(k, "")).strip():
            raise ValueError(f"EDF+ patient {k} may not contain a space: {p.get(k)!r}")
    return " ".join([_subfield(p.get("code", "")), _subfield(p.get("sex", "")),
                     _subfield(p.get("birthdate", "")), _subfield(p.get("name", ""))])


def edfplus_recording(startdate: str) -> str:
    return f"Startdate {edfplus_date(startdate)} X X X"


# ---------------------------------------------------------------- TALs
def _fmt(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{float(x):.6f}".rstrip("0").rstrip(".")


def _tal(onset: float, duration: float | None, texts: list[str]) -> bytes:
    s = ("+" if onset >= 0 else "-") + _fmt(abs(onset))
    if duration:
        s += "\x15" + _fmt(duration)
    s += "\x14" + "".join(t + "\x14" for t in texts)
    if not texts:
        s += "\x14"
    return s.encode("utf-8") + b"\x00"


def _parse_tals(block: bytes) -> list[tuple[float, float, list[str]]]:
    out = []
    for raw in block.split(b"\x00"):
        if not raw:
            continue
        parts = raw.decode("utf-8").split("\x14")
        head, texts = parts[0], [t for t in parts[1:] if t != ""]
        onset_s, _, dur_s = head.partition("\x15")
        out.append((float(onset_s), float(dur_s) if dur_s else 0.0, texts))
    return out


# ---------------------------------------------------------------- write
def write_edf(src: str | pathlib.Path, dst: str | pathlib.Path, *, patient: str, recording: str,
              annotations: list[dict] | None, edfplus: bool) -> dict:
    """Copy `src` to `dst` with new identification fields; `edfplus=True` also appends the
    annotation signal carrying `annotations` ([{onset_sec, duration_sec, text}], may be empty).
    Returns the file's metadata (`recording_meta`)."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    h = read_header(src)
    if h.edfplus or any(s["label"] == ANNOT_LABEL for s in h.signals):
        raise ValueError(f"{src}: already EDF+; only plain EDF cuts are rewritten")
    for name, v in (("patient", patient), ("recording", recording)):
        if len(v) > 80 or not v.isascii():
            raise ValueError(f"{name} field must be ASCII and at most 80 characters: {v!r}")
    blob = src.read_bytes()
    data = blob[h.header_bytes:]
    if len(data) != h.record_bytes * h.n_records:
        raise ValueError(f"{src}: data length {len(data)} != {h.n_records} records x {h.record_bytes} bytes")
    main = bytearray(blob[:256])
    main[8:88] = patient.ljust(80).encode("ascii")
    main[88:168] = recording.ljust(80).encode("ascii")
    if not edfplus:
        if annotations:
            raise ValueError("annotations need EDF+ (edfplus=True)")
        dst.write_bytes(bytes(main) + blob[256:])
        return recording_meta(dst)
    anns = sorted(annotations or [], key=lambda a: (float(a["onset_sec"]), str(a["text"])))
    total = h.n_records * h.record_duration
    per_record: list[list[bytes]] = [[] for _ in range(h.n_records)]
    for a in anns:
        on = float(a["onset_sec"])
        if not 0 <= on < total:
            raise ValueError(f"annotation onset {on} outside the recording [0, {total})")
        r = min(int(on // h.record_duration), h.n_records - 1)
        per_record[r].append(_tal(on, float(a.get("duration_sec") or 0), [str(a["text"])]))
    blocks = [_tal(r * h.record_duration, None, []) + b"".join(per_record[r]) for r in range(h.n_records)]
    n_samples = max(8, math.ceil(max(len(b) for b in blocks) / 2))
    width = 2 * n_samples
    ns = h.ns + 1
    main[184:192] = str(256 * (ns + 1)).ljust(8).encode("ascii")
    main[192:236] = "EDF+C".ljust(44).encode("ascii")
    main[252:256] = str(ns).ljust(4).encode("ascii")
    annot_values = {"label": ANNOT_LABEL, "transducer": "", "physical_dimension": "", "physical_min": "-1",
                    "physical_max": "1", "digital_min": "-32768", "digital_max": "32767", "prefilter": "",
                    "samples": str(n_samples), "reserved": ""}
    sig = b""
    for k, (name, w) in enumerate(_SIGNAL_FIELDS):
        sig += b"".join(h.raw_signal_fields[k]) + annot_values[name].ljust(w).encode("ascii")
    rec = h.record_bytes
    out = bytearray(bytes(main) + sig)
    for r in range(h.n_records):
        out += data[r * rec:(r + 1) * rec] + blocks[r].ljust(width, b"\x00")
    dst.write_bytes(bytes(out))
    meta = recording_meta(dst)
    if meta["annotations"] != [{"onset_sec": float(a["onset_sec"]), "duration_sec": float(a.get("duration_sec") or 0),
                                "text": str(a["text"])} for a in anns]:
        raise AssertionError(f"{dst}: annotations do not read back")
    return meta


# ---------------------------------------------------------------- read back
def _annotation_blocks(path: pathlib.Path) -> list[bytes]:
    h = read_header(path)
    idx = [i for i, s in enumerate(h.signals) if s["label"] == ANNOT_LABEL]
    if not idx:
        return []
    offs = [0]
    for s in h.signals:
        offs.append(offs[-1] + 2 * s["samples"])
    i = idx[0]
    blob = pathlib.Path(path).read_bytes()[h.header_bytes:]
    rec = h.record_bytes
    return [blob[r * rec + offs[i]: r * rec + offs[i + 1]] for r in range(h.n_records)]


def read_timekeeping(path: str | pathlib.Path) -> list[float]:
    """The onset of every data record, from its first TAL."""
    return [(_parse_tals(b) or [(math.nan, 0, [])])[0][0] for b in _annotation_blocks(pathlib.Path(path))]


def read_annotations(path: str | pathlib.Path) -> list[dict]:
    """Every annotation TAL of the file (timekeeping TALs excluded), in file order."""
    out = []
    for b in _annotation_blocks(pathlib.Path(path)):
        for k, (on, du, texts) in enumerate(_parse_tals(b)):
            if k == 0 and not texts:
                continue                          # timekeeping TAL
            for t in texts:
                out.append({"onset_sec": on, "duration_sec": du, "text": t})
    return out


def most_common_rate(h: Header) -> float:
    """Most common sampling rate over the data signals; ties go to the higher rate."""
    rates = Counter(round(s["samples"] / h.record_duration, 6) for s in h.data_signals())
    best = max(rates.items(), key=lambda kv: (kv[1], kv[0]))
    return float(best[0])


def recording_meta(path: str | pathlib.Path) -> dict:
    """What the gold records about one file: format, duration, channel count (annotation signal
    excluded), most common rate and the annotations the file itself carries."""
    h = read_header(path)
    return {"format": "EDF+" if h.edfplus else "EDF",
            "duration_sec": float(h.n_records * h.record_duration),
            "channel_count": len(h.data_signals()),
            "sfreq_hz": most_common_rate(h),
            "annotations": read_annotations(path) if h.edfplus else []}
