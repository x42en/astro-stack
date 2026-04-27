"""Static catalogue of popular deep-sky objects.

This is a curated, offline-first reference table covering the full Messier
catalogue (M1–M110) plus a small selection of frequently photographed NGC/IC
targets. Coordinates are J2000 (decimal degrees) and were sourced from the
SIMBAD master tables; magnitudes and types come from the same source where
available.

The table is bundled with the application so the front-end target picker
works without any network access — SIMBAD resolution is layered on top via
``app/api/v1/catalog.py`` for objects not present here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CatalogObject:
    """A single deep-sky object entry.

    Attributes:
        id: Stable identifier used by the front-end (e.g. ``"M42"``).
        name: Common name (e.g. ``"Orion Nebula"``).
        type: Short object type code (``nebula``, ``galaxy``, ``cluster``,
            ``planetary``, ``supernova``, ``other``).
        constellation: IAU constellation abbreviation (e.g. ``"Ori"``).
        ra_deg: Right ascension, J2000 decimal degrees.
        dec_deg: Declination, J2000 decimal degrees.
        magnitude: Visual magnitude (lower = brighter); ``None`` if unknown.
    """

    id: str
    name: str
    type: str
    constellation: str
    ra_deg: float
    dec_deg: float
    magnitude: float | None = None


# fmt: off
# (id, name, type, constellation, ra_deg, dec_deg, magnitude)
_MESSIER: tuple[CatalogObject, ...] = (
    CatalogObject("M1",   "Crab Nebula",                  "supernova",  "Tau",   83.6331,  22.0145, 8.4),
    CatalogObject("M2",   "Globular Cluster M2",          "cluster",    "Aqr",  323.3625,  -0.8233, 6.5),
    CatalogObject("M3",   "Globular Cluster M3",          "cluster",    "CVn",  205.5483,  28.3772, 6.2),
    CatalogObject("M4",   "Globular Cluster M4",          "cluster",    "Sco",  245.8967, -26.5258, 5.9),
    CatalogObject("M5",   "Globular Cluster M5",          "cluster",    "Ser",  229.6383,   2.0810, 5.6),
    CatalogObject("M6",   "Butterfly Cluster",            "cluster",    "Sco",  265.0833, -32.2167, 4.2),
    CatalogObject("M7",   "Ptolemy's Cluster",            "cluster",    "Sco",  268.4500, -34.7833, 3.3),
    CatalogObject("M8",   "Lagoon Nebula",                "nebula",     "Sgr",  270.9042, -24.3867, 6.0),
    CatalogObject("M9",   "Globular Cluster M9",          "cluster",    "Oph",  259.7992, -18.5161, 7.7),
    CatalogObject("M10",  "Globular Cluster M10",         "cluster",    "Oph",  254.2875,  -4.1003, 6.6),
    CatalogObject("M11",  "Wild Duck Cluster",            "cluster",    "Sct",  282.7708,  -6.2700, 6.3),
    CatalogObject("M12",  "Globular Cluster M12",         "cluster",    "Oph",  251.8092,  -1.9483, 6.7),
    CatalogObject("M13",  "Hercules Globular Cluster",    "cluster",    "Her",  250.4233,  36.4611, 5.8),
    CatalogObject("M14",  "Globular Cluster M14",         "cluster",    "Oph",  264.4008,  -3.2458, 7.6),
    CatalogObject("M15",  "Globular Cluster M15",         "cluster",    "Peg",  322.4933,  12.1670, 6.2),
    CatalogObject("M16",  "Eagle Nebula",                 "nebula",     "Ser",  274.7000, -13.8067, 6.4),
    CatalogObject("M17",  "Omega Nebula",                 "nebula",     "Sgr",  275.1958, -16.1717, 6.0),
    CatalogObject("M18",  "Open Cluster M18",             "cluster",    "Sgr",  274.9917, -17.1000, 7.5),
    CatalogObject("M19",  "Globular Cluster M19",         "cluster",    "Oph",  255.6575, -26.2678, 6.8),
    CatalogObject("M20",  "Trifid Nebula",                "nebula",     "Sgr",  270.6750, -23.0333, 6.3),
    CatalogObject("M21",  "Open Cluster M21",             "cluster",    "Sgr",  271.0500, -22.5000, 6.5),
    CatalogObject("M22",  "Sagittarius Cluster",          "cluster",    "Sgr",  279.0997, -23.9047, 5.1),
    CatalogObject("M23",  "Open Cluster M23",             "cluster",    "Sgr",  269.2667, -18.9833, 6.9),
    CatalogObject("M24",  "Sagittarius Star Cloud",       "other",      "Sgr",  274.2000, -18.5500, 4.6),
    CatalogObject("M25",  "Open Cluster M25",             "cluster",    "Sgr",  277.9417, -19.1167, 6.5),
    CatalogObject("M26",  "Open Cluster M26",             "cluster",    "Sct",  281.3208,  -9.3833, 8.0),
    CatalogObject("M27",  "Dumbbell Nebula",              "planetary",  "Vul",  299.9017,  22.7211, 7.5),
    CatalogObject("M28",  "Globular Cluster M28",         "cluster",    "Sgr",  276.1370, -24.8694, 6.8),
    CatalogObject("M29",  "Open Cluster M29",             "cluster",    "Cyg",  305.9750,  38.5333, 7.1),
    CatalogObject("M30",  "Globular Cluster M30",         "cluster",    "Cap",  325.0925, -23.1798, 7.2),
    CatalogObject("M31",  "Andromeda Galaxy",             "galaxy",     "And",   10.6847,  41.2692, 3.4),
    CatalogObject("M32",  "Galaxy M32",                   "galaxy",     "And",   10.6743,  40.8652, 8.1),
    CatalogObject("M33",  "Triangulum Galaxy",            "galaxy",     "Tri",   23.4621,  30.6602, 5.7),
    CatalogObject("M34",  "Open Cluster M34",             "cluster",    "Per",   40.5333,  42.7833, 5.5),
    CatalogObject("M35",  "Open Cluster M35",             "cluster",    "Gem",   92.2250,  24.3333, 5.3),
    CatalogObject("M36",  "Pinwheel Cluster",             "cluster",    "Aur",   84.0833,  34.1333, 6.3),
    CatalogObject("M37",  "Open Cluster M37",             "cluster",    "Aur",   88.0750,  32.5500, 6.2),
    CatalogObject("M38",  "Starfish Cluster",             "cluster",    "Aur",   82.1667,  35.8333, 7.4),
    CatalogObject("M39",  "Open Cluster M39",             "cluster",    "Cyg",  322.8750,  48.4333, 5.5),
    CatalogObject("M40",  "Winnecke 4",                   "other",      "UMa",  185.5500,  58.0833, 9.7),
    CatalogObject("M41",  "Open Cluster M41",             "cluster",    "CMa",  101.5042, -20.7167, 4.5),
    CatalogObject("M42",  "Orion Nebula",                 "nebula",     "Ori",   83.8221,  -5.3911, 4.0),
    CatalogObject("M43",  "De Mairan's Nebula",           "nebula",     "Ori",   83.8792,  -5.2700, 9.0),
    CatalogObject("M44",  "Beehive Cluster",              "cluster",    "Cnc",  130.0250,  19.6833, 3.7),
    CatalogObject("M45",  "Pleiades",                     "cluster",    "Tau",   56.7500,  24.1167, 1.6),
    CatalogObject("M46",  "Open Cluster M46",             "cluster",    "Pup",  115.4458, -14.8167, 6.1),
    CatalogObject("M47",  "Open Cluster M47",             "cluster",    "Pup",  114.1458, -14.4833, 4.4),
    CatalogObject("M48",  "Open Cluster M48",             "cluster",    "Hya",  123.4292,  -5.7500, 5.8),
    CatalogObject("M49",  "Galaxy M49",                   "galaxy",     "Vir",  187.4448,   8.0004, 8.4),
    CatalogObject("M50",  "Open Cluster M50",             "cluster",    "Mon",  105.7000,  -8.3333, 5.9),
    CatalogObject("M51",  "Whirlpool Galaxy",             "galaxy",     "CVn",  202.4696,  47.1952, 8.4),
    CatalogObject("M52",  "Open Cluster M52",             "cluster",    "Cas",  351.2000,  61.5833, 5.0),
    CatalogObject("M53",  "Globular Cluster M53",         "cluster",    "Com",  198.2304,  18.1683, 7.6),
    CatalogObject("M54",  "Globular Cluster M54",         "cluster",    "Sgr",  283.7637, -30.4799, 7.6),
    CatalogObject("M55",  "Globular Cluster M55",         "cluster",    "Sgr",  294.9988, -30.9648, 6.3),
    CatalogObject("M56",  "Globular Cluster M56",         "cluster",    "Lyr",  289.1483,  30.1834, 8.3),
    CatalogObject("M57",  "Ring Nebula",                  "planetary",  "Lyr",  283.3962,  33.0292, 8.8),
    CatalogObject("M58",  "Galaxy M58",                   "galaxy",     "Vir",  189.4313,  11.8181, 9.7),
    CatalogObject("M59",  "Galaxy M59",                   "galaxy",     "Vir",  190.5096,  11.6470, 9.6),
    CatalogObject("M60",  "Galaxy M60",                   "galaxy",     "Vir",  190.9167,  11.5526, 8.8),
    CatalogObject("M61",  "Galaxy M61",                   "galaxy",     "Vir",  185.4790,   4.4737, 9.7),
    CatalogObject("M62",  "Globular Cluster M62",         "cluster",    "Oph",  255.3025, -30.1124, 6.5),
    CatalogObject("M63",  "Sunflower Galaxy",             "galaxy",     "CVn",  198.9554,  42.0293, 8.6),
    CatalogObject("M64",  "Black Eye Galaxy",             "galaxy",     "Com",  194.1820,  21.6826, 8.5),
    CatalogObject("M65",  "Galaxy M65",                   "galaxy",     "Leo",  169.7333,  13.0922, 9.3),
    CatalogObject("M66",  "Galaxy M66",                   "galaxy",     "Leo",  170.0625,  12.9912, 8.9),
    CatalogObject("M67",  "Open Cluster M67",             "cluster",    "Cnc",  132.8250,  11.8000, 6.1),
    CatalogObject("M68",  "Globular Cluster M68",         "cluster",    "Hya",  189.8665, -26.7448, 7.8),
    CatalogObject("M69",  "Globular Cluster M69",         "cluster",    "Sgr",  277.8458, -32.3481, 7.6),
    CatalogObject("M70",  "Globular Cluster M70",         "cluster",    "Sgr",  280.8038, -32.2921, 7.9),
    CatalogObject("M71",  "Globular Cluster M71",         "cluster",    "Sge",  298.4438,  18.7791, 8.2),
    CatalogObject("M72",  "Globular Cluster M72",         "cluster",    "Aqr",  313.3654, -12.5374, 9.3),
    CatalogObject("M73",  "Asterism M73",                 "other",      "Aqr",  314.7500, -12.6333, 9.0),
    CatalogObject("M74",  "Phantom Galaxy",               "galaxy",     "Psc",   24.1742,  15.7836, 9.4),
    CatalogObject("M75",  "Globular Cluster M75",         "cluster",    "Sgr",  301.5202, -21.9226, 8.5),
    CatalogObject("M76",  "Little Dumbbell Nebula",       "planetary",  "Per",   25.5821,  51.5754, 10.1),
    CatalogObject("M77",  "Cetus A",                      "galaxy",     "Cet",   40.6696,  -0.0133, 8.9),
    CatalogObject("M78",  "Reflection Nebula M78",        "nebula",     "Ori",   86.6917,   0.0500, 8.3),
    CatalogObject("M79",  "Globular Cluster M79",         "cluster",    "Lep",   81.0438, -24.5244, 7.7),
    CatalogObject("M80",  "Globular Cluster M80",         "cluster",    "Sco",  244.2600, -22.9764, 7.3),
    CatalogObject("M81",  "Bode's Galaxy",                "galaxy",     "UMa",  148.8882,  69.0653, 6.9),
    CatalogObject("M82",  "Cigar Galaxy",                 "galaxy",     "UMa",  148.9696,  69.6797, 8.4),
    CatalogObject("M83",  "Southern Pinwheel Galaxy",     "galaxy",     "Hya",  204.2538, -29.8654, 7.5),
    CatalogObject("M84",  "Galaxy M84",                   "galaxy",     "Vir",  186.2656,  12.8870, 9.1),
    CatalogObject("M85",  "Galaxy M85",                   "galaxy",     "Com",  186.3504,  18.1912, 9.1),
    CatalogObject("M86",  "Galaxy M86",                   "galaxy",     "Vir",  186.5490,  12.9462, 8.9),
    CatalogObject("M87",  "Virgo A",                      "galaxy",     "Vir",  187.7059,  12.3911, 8.6),
    CatalogObject("M88",  "Galaxy M88",                   "galaxy",     "Com",  187.9966,  14.4204, 9.6),
    CatalogObject("M89",  "Galaxy M89",                   "galaxy",     "Vir",  188.9159,  12.5563, 9.8),
    CatalogObject("M90",  "Galaxy M90",                   "galaxy",     "Vir",  189.2076,  13.1626, 9.5),
    CatalogObject("M91",  "Galaxy M91",                   "galaxy",     "Com",  188.8600,  14.4960, 10.2),
    CatalogObject("M92",  "Globular Cluster M92",         "cluster",    "Her",  259.2808,  43.1359, 6.4),
    CatalogObject("M93",  "Open Cluster M93",             "cluster",    "Pup",  116.1417, -23.8667, 6.2),
    CatalogObject("M94",  "Croc's Eye Galaxy",            "galaxy",     "CVn",  192.7211,  41.1203, 8.2),
    CatalogObject("M95",  "Galaxy M95",                   "galaxy",     "Leo",  160.9908,  11.7038, 9.7),
    CatalogObject("M96",  "Galaxy M96",                   "galaxy",     "Leo",  161.6906,  11.8200, 9.2),
    CatalogObject("M97",  "Owl Nebula",                   "planetary",  "UMa",  168.6987,  55.0190, 9.9),
    CatalogObject("M98",  "Galaxy M98",                   "galaxy",     "Com",  183.4513,  14.9003, 10.1),
    CatalogObject("M99",  "Coma Pinwheel Galaxy",         "galaxy",     "Com",  184.7067,  14.4163, 9.9),
    CatalogObject("M100", "Galaxy M100",                  "galaxy",     "Com",  185.7288,  15.8223, 9.3),
    CatalogObject("M101", "Pinwheel Galaxy",              "galaxy",     "UMa",  210.8025,  54.3486, 7.9),
    CatalogObject("M102", "Spindle Galaxy",               "galaxy",     "Dra",  226.6231,  55.7634, 9.9),
    CatalogObject("M103", "Open Cluster M103",            "cluster",    "Cas",   23.3417,  60.6500, 7.4),
    CatalogObject("M104", "Sombrero Galaxy",              "galaxy",     "Vir",  189.9976, -11.6231, 8.0),
    CatalogObject("M105", "Galaxy M105",                  "galaxy",     "Leo",  161.9566,  12.5816, 9.3),
    CatalogObject("M106", "Galaxy M106",                  "galaxy",     "CVn",  184.7396,  47.3037, 8.4),
    CatalogObject("M107", "Globular Cluster M107",        "cluster",    "Oph",  248.1325, -13.0537, 7.8),
    CatalogObject("M108", "Surfboard Galaxy",             "galaxy",     "UMa",  167.8792,  55.6741, 10.0),
    CatalogObject("M109", "Galaxy M109",                  "galaxy",     "UMa",  179.3996,  53.3745, 9.8),
    CatalogObject("M110", "Galaxy M110",                  "galaxy",     "And",   10.0917,  41.6852, 8.1),
)

_OTHER: tuple[CatalogObject, ...] = (
    CatalogObject("NGC7000", "North America Nebula",      "nebula",     "Cyg",  314.7500,  44.3333, 4.0),
    CatalogObject("IC1396",  "Elephant's Trunk Nebula",   "nebula",     "Cep",  324.6667,  57.5000, 3.5),
    CatalogObject("NGC891",  "Silver Sliver Galaxy",      "galaxy",     "And",   35.6396,  42.3492, 9.9),
    CatalogObject("NGC2244", "Rosette Nebula Cluster",    "nebula",     "Mon",   97.9833,   4.9333, 4.8),
    CatalogObject("NGC2264", "Cone / Christmas Tree",     "nebula",     "Mon",  100.2417,   9.8833, 3.9),
    CatalogObject("NGC6960", "Western Veil Nebula",       "nebula",     "Cyg",  312.7500,  30.7000, 7.0),
    CatalogObject("NGC6992", "Eastern Veil Nebula",       "nebula",     "Cyg",  313.0000,  31.7167, 7.0),
    CatalogObject("NGC7293", "Helix Nebula",              "planetary",  "Aqr",  337.4108, -20.8372, 7.6),
    CatalogObject("NGC7635", "Bubble Nebula",             "nebula",     "Cas",  350.2042,  61.2000, 10.0),
    CatalogObject("IC434",   "Horsehead Nebula",          "nebula",     "Ori",   85.2417,  -2.4583, 6.8),
    CatalogObject("NGC2024", "Flame Nebula",              "nebula",     "Ori",   85.4292,  -1.8500, 2.0),
    CatalogObject("NGC1499", "California Nebula",         "nebula",     "Per",   60.2917,  36.4167, 5.0),
    CatalogObject("NGC281",  "Pacman Nebula",             "nebula",     "Cas",   13.0000,  56.6167, 7.4),
    CatalogObject("NGC4565", "Needle Galaxy",             "galaxy",     "Com",  189.0867,  25.9875, 9.6),
    CatalogObject("IC1805",  "Heart Nebula",              "nebula",     "Cas",   38.7500,  61.4500, 6.5),
    CatalogObject("IC1848",  "Soul Nebula",               "nebula",     "Cas",   42.6667,  60.4500, 6.5),
)
# fmt: on


_ALL_OBJECTS: tuple[CatalogObject, ...] = _MESSIER + _OTHER
_BY_ID: dict[str, CatalogObject] = {obj.id.upper(): obj for obj in _ALL_OBJECTS}


def all_objects() -> Iterable[CatalogObject]:
    """Return every catalogued object in stable display order."""
    return _ALL_OBJECTS


def get_by_id(object_id: str) -> CatalogObject | None:
    """Look up a single catalogue entry by its identifier (case-insensitive).

    Args:
        object_id: Catalogue identifier such as ``"M42"`` or ``"NGC7000"``.

    Returns:
        The matching :class:`CatalogObject`, or ``None`` if not found.
    """
    return _BY_ID.get(object_id.upper().replace(" ", ""))


def search(query: str, limit: int = 50) -> list[CatalogObject]:
    """Substring + token search across id, name and constellation.

    Matching is case-insensitive and tolerant of whitespace. Results are
    ordered with id-prefix matches first, then name-prefix matches, then
    everything else, and capped at ``limit``.

    Args:
        query: Free-text query (id, name fragment, constellation).
        limit: Maximum number of results.

    Returns:
        List of matching :class:`CatalogObject`, sorted by relevance.
    """
    q = query.strip().lower().replace(" ", "")
    if not q:
        return list(_ALL_OBJECTS[:limit])

    id_prefix: list[CatalogObject] = []
    name_prefix: list[CatalogObject] = []
    other: list[CatalogObject] = []

    for obj in _ALL_OBJECTS:
        oid = obj.id.lower()
        name = obj.name.lower()
        const = obj.constellation.lower()
        if oid.startswith(q):
            id_prefix.append(obj)
        elif name.startswith(q) or name.replace(" ", "").startswith(q):
            name_prefix.append(obj)
        elif q in oid or q in name.replace(" ", "") or q == const:
            other.append(obj)

    return (id_prefix + name_prefix + other)[:limit]
