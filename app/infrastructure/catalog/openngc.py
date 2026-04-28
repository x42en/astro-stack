"""Curated NGC and IC deep-sky objects for amateur astrophotography.

A selection of well-known NGC/IC objects suitable for telescope observation.
Coordinates are J2000 (decimal degrees) at roughly 1' precision; magnitudes
are V-band when known.

Inspired by the OpenNGC project (cdsweb.u-strasbg.fr) — CC BY-SA.
"""

from __future__ import annotations

from app.infrastructure.catalog.messier import CatalogObject

# fmt: off
_OPEN_NGC: tuple[CatalogObject, ...] = (
    # ── Galaxies ────────────────────────────────────────────────────────────
    CatalogObject("NGC253",  "Sculptor Galaxy",       "galaxy", "Scl",  11.8875, -25.2917, 7.1),
    CatalogObject("NGC891",  "Silver Sliver Galaxy",  "galaxy", "And",  35.6396,  42.3492, 9.9),
    CatalogObject("NGC1232", "NGC 1232",              "galaxy", "Eri",  47.4375, -20.5792, 9.8),
    CatalogObject("NGC1300", "Barred Spiral 1300",    "galaxy", "Eri",  49.9208, -19.4111, 10.4),
    CatalogObject("NGC1365", "Great Barred Spiral",   "galaxy", "For",  53.4017, -36.1403, 9.6),
    CatalogObject("NGC2403", "NGC 2403",              "galaxy", "Cam", 114.2125,  65.6028, 8.9),
    CatalogObject("NGC3344", "NGC 3344",              "galaxy", "LMi", 160.8792,  24.9222, 10.4),
    CatalogObject("NGC3628", "Hamburger Galaxy",      "galaxy", "Leo", 170.0708,  13.5894, 9.7),
    CatalogObject("NGC4565", "Needle Galaxy",         "galaxy", "Com", 189.0867,  25.9875, 9.6),
    CatalogObject("NGC4631", "Whale Galaxy",          "galaxy", "CVn", 190.5333,  32.5417, 9.3),
    CatalogObject("NGC4945", "NGC 4945",              "galaxy", "Cen", 196.3625, -49.4683, 8.6),
    CatalogObject("NGC5005", "NGC 5005",              "galaxy", "CVn", 197.7333,  37.0589, 9.8),
    CatalogObject("NGC5907", "Splinter Galaxy",       "galaxy", "Dra", 228.9750,  56.3289, 10.4),
    CatalogObject("NGC6946", "Fireworks Galaxy",      "galaxy", "Cep", 308.7180,  60.1539, 9.6),
    CatalogObject("NGC7331", "NGC 7331",              "galaxy", "Peg", 339.2671,  34.4156, 9.5),
    CatalogObject("NGC7479", "Propeller Galaxy",      "galaxy", "Peg", 346.2354,  12.3225, 10.9),
    CatalogObject("NGC7814", "Little Sombrero",       "galaxy", "Peg",   0.8125,  16.1456, 10.6),

    # ── Diffuse / emission nebulae ──────────────────────────────────────────
    CatalogObject("NGC281",  "Pacman Nebula",         "nebula", "Cas",  13.0000,  56.6167, 7.4),
    CatalogObject("NGC1499", "California Nebula",     "nebula", "Per",  60.2917,  36.4167, 5.0),
    CatalogObject("NGC2174", "Monkey Head Nebula",    "nebula", "Ori",  91.6667,  20.5000, 6.8),
    CatalogObject("NGC6357", "Lobster Nebula",        "nebula", "Sco", 261.4500, -34.1833, 6.0),
    CatalogObject("NGC6888", "Crescent Nebula",       "nebula", "Cyg", 303.0167,  38.3500, 7.4),
    CatalogObject("NGC6960", "Western Veil Nebula",   "nebula", "Cyg", 312.7500,  30.7000, 7.0),
    CatalogObject("NGC6992", "Eastern Veil Nebula",   "nebula", "Cyg", 313.0000,  31.7167, 7.0),
    CatalogObject("NGC7000", "North America Nebula",  "nebula", "Cyg", 314.7500,  44.3333, 4.0),
    CatalogObject("NGC7023", "Iris Nebula",           "nebula", "Cep", 315.3917,  68.1639, 6.8),
    CatalogObject("NGC7129", "NGC 7129",              "nebula", "Cep", 325.7833,  66.1167, 11.5),
    CatalogObject("NGC7635", "Bubble Nebula",         "nebula", "Cas", 350.2042,  61.2000, 10.0),
    CatalogObject("NGC7822", "NGC 7822",              "nebula", "Cep",   0.2375,  68.5833, None),
    CatalogObject("IC405",   "Flaming Star Nebula",   "nebula", "Aur",  79.0833,  34.4500, 6.0),
    CatalogObject("IC410",   "Tadpoles Nebula",       "nebula", "Aur",  80.7250,  33.4667, 7.5),
    CatalogObject("IC434",   "Horsehead Nebula",      "nebula", "Ori",  85.2417,  -2.4583, 6.8),
    CatalogObject("IC443",   "Jellyfish Nebula",      "nebula", "Gem",  94.2583,  22.5333, 12.0),
    CatalogObject("IC1318",  "Sadr Region",           "nebula", "Cyg", 305.5667,  40.4500, None),
    CatalogObject("IC1396",  "Elephant's Trunk",      "nebula", "Cep", 324.6667,  57.5000, 3.5),
    CatalogObject("IC1805",  "Heart Nebula",          "nebula", "Cas",  38.7500,  61.4500, 6.5),
    CatalogObject("IC1848",  "Soul Nebula",           "nebula", "Cas",  42.6667,  60.4500, 6.5),
    CatalogObject("IC2118",  "Witch Head Nebula",     "nebula", "Eri",  77.3333,  -7.1167, 13.0),
    CatalogObject("IC5070",  "Pelican Nebula",        "nebula", "Cyg", 314.8208,  44.3500, 8.0),
    CatalogObject("IC5146",  "Cocoon Nebula",         "nebula", "Cyg", 328.3833,  47.2667, 7.2),

    # ── Open clusters ───────────────────────────────────────────────────────
    CatalogObject("NGC457",  "Owl Cluster",           "cluster","Cas",  19.0833,  58.2833, 6.4),
    CatalogObject("NGC663",  "NGC 663",               "cluster","Cas",  26.2500,  61.2167, 7.1),
    CatalogObject("NGC869",  "Double Cluster West",   "cluster","Per",  34.7417,  57.1333, 5.3),
    CatalogObject("NGC884",  "Double Cluster East",   "cluster","Per",  35.5417,  57.1500, 6.1),
    CatalogObject("NGC1502", "Kemble's Cluster",      "cluster","Cam",  62.0292,  62.3267, 5.7),
    CatalogObject("NGC2244", "Rosette Cluster",       "cluster","Mon",  97.9833,   4.9333, 4.8),
    CatalogObject("NGC2264", "Christmas Tree Cluster","cluster","Mon", 100.2417,   9.8833, 3.9),
    CatalogObject("NGC2362", "Tau CMa Cluster",       "cluster","CMa", 109.6708, -24.9572, 4.1),
    CatalogObject("NGC6231", "Northern Jewel Box",    "cluster","Sco", 253.5417, -41.8267, 2.6),
    CatalogObject("NGC7789", "Caroline's Rose",       "cluster","Cas", 359.3208,  56.7167, 6.7),

    # ── Globular clusters ───────────────────────────────────────────────────
    CatalogObject("NGC288",  "NGC 288",               "cluster","Scl",  13.1880, -26.5828, 8.1),
    CatalogObject("NGC5139", "Omega Centauri",        "cluster","Cen", 201.6917, -47.4792, 3.7),
    CatalogObject("NGC6397", "NGC 6397",              "cluster","Ara", 265.1750, -53.6744, 5.2),
    CatalogObject("NGC6752", "NGC 6752",              "cluster","Pav", 287.7167, -59.9844, 5.4),

    # ── Planetary nebulae ───────────────────────────────────────────────────
    CatalogObject("NGC246",  "Skull Nebula",          "planetary","Cet", 11.7625, -11.8728, 8.0),
    CatalogObject("NGC2392", "Eskimo Nebula",         "planetary","Gem", 112.2917, 20.9117, 9.2),
    CatalogObject("NGC3242", "Ghost of Jupiter",      "planetary","Hya", 156.0083,-18.6275, 8.6),
    CatalogObject("NGC6543", "Cat's Eye Nebula",      "planetary","Dra", 269.6383, 66.6333, 8.1),
    CatalogObject("NGC6826", "Blinking Planetary",    "planetary","Cyg", 296.2167, 50.5258, 8.8),
    CatalogObject("NGC7009", "Saturn Nebula",         "planetary","Aqr", 316.0417,-11.3631, 8.0),
    CatalogObject("NGC7293", "Helix Nebula",          "planetary","Aqr", 337.4108,-20.8372, 7.6),
    CatalogObject("NGC7662", "Blue Snowball",         "planetary","And", 351.4500, 42.5392, 8.3),

    # ── Supernova remnants ──────────────────────────────────────────────────
    CatalogObject("NGC6995", "Eastern Veil SNR",      "supernova","Cyg", 313.1167, 31.4333, 7.0),
)
# fmt: on


def all_openngc() -> tuple[CatalogObject, ...]:
    """Return the curated NGC/IC catalog."""
    return _OPEN_NGC
