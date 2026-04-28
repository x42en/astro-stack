"""Curated list of popular and beginner-friendly deep-sky targets.

Each :class:`PopularEntry` references a canonical ``catalog_id`` that exists
in the unified registry (Messier or OpenNGC), and adds friendly metadata for
the session-planning UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PopularEntry:
    """Marketing/UX metadata for a popular DSO target.

    Attributes:
        catalog_id: Canonical identifier resolvable by the registry.
        friendly_name: Human-readable marketing name.
        difficulty: One of ``beginner``, ``intermediate``, ``advanced``.
        season: One of ``winter``, ``spring``, ``summer``, ``autumn``,
            ``all-year``.
        blurb: One-sentence why-it's-cool description.
    """

    catalog_id: str
    friendly_name: str
    difficulty: str
    season: str
    blurb: str


POPULAR: tuple[PopularEntry, ...] = (
    PopularEntry("M31", "Andromeda Galaxy", "beginner", "autumn",
                 "Nearest large galaxy; spans 6× the full moon."),
    PopularEntry("M42", "Orion Nebula", "beginner", "winter",
                 "Brightest emission nebula in the sky; visible to the naked eye."),
    PopularEntry("M45", "Pleiades", "beginner", "winter",
                 "Brilliant young star cluster; jewel of winter skies."),
    PopularEntry("M51", "Whirlpool Galaxy", "intermediate", "spring",
                 "Iconic face-on spiral with a clear interacting companion."),
    PopularEntry("M81", "Bode's Galaxy", "intermediate", "spring",
                 "Bright spiral paired with M82 in the same eyepiece field."),
    PopularEntry("M101", "Pinwheel Galaxy", "intermediate", "spring",
                 "Face-on spiral with delicate arms; needs dark skies."),
    PopularEntry("NGC7000", "North America Nebula", "intermediate", "summer",
                 "Huge emission cloud shaped like the continent."),
    PopularEntry("IC1396", "Elephant's Trunk Nebula", "intermediate", "autumn",
                 "Sculpted dust pillars inside a wide H-alpha region."),
    PopularEntry("IC1805", "Heart Nebula", "intermediate", "autumn",
                 "Vast emission nebula forming a heart shape in Cassiopeia."),
    PopularEntry("NGC869", "Double Cluster", "beginner", "autumn",
                 "Two brilliant open clusters side-by-side; binocular favourite."),
    PopularEntry("NGC2244", "Rosette Cluster", "intermediate", "winter",
                 "Open cluster embedded in the rose-shaped Rosette Nebula."),
    PopularEntry("M27", "Dumbbell Nebula", "intermediate", "summer",
                 "Brightest planetary nebula; clear dumbbell shape."),
    PopularEntry("M57", "Ring Nebula", "intermediate", "summer",
                 "Iconic ring-shaped planetary nebula in Lyra."),
    PopularEntry("M97", "Owl Nebula", "advanced", "spring",
                 "Round planetary with two darker eye-like patches."),
    PopularEntry("M104", "Sombrero Galaxy", "intermediate", "spring",
                 "Edge-on spiral with a striking dark dust lane."),
    PopularEntry("M13", "Hercules Cluster", "beginner", "summer",
                 "Brightest globular cluster of the northern sky."),
    PopularEntry("M22", "Sagittarius Cluster", "beginner", "summer",
                 "Brilliant globular cluster easy to resolve in small scopes."),
    PopularEntry("NGC6960", "Western Veil", "advanced", "summer",
                 "Delicate supernova remnant; OIII filter recommended."),
    PopularEntry("NGC6992", "Eastern Veil", "advanced", "summer",
                 "Companion arc of the Veil supernova remnant."),
    PopularEntry("NGC7293", "Helix Nebula", "advanced", "autumn",
                 "Largest and nearest planetary nebula; needs dark skies."),
    PopularEntry("NGC4565", "Needle Galaxy", "intermediate", "spring",
                 "Magnificent edge-on spiral with a sharp dust lane."),
    PopularEntry("M33", "Triangulum Galaxy", "intermediate", "autumn",
                 "Third-largest galaxy in the Local Group; faint but huge."),
    PopularEntry("M3", "M3 Globular Cluster", "intermediate", "spring",
                 "Dense northern globular; resolves easily in 4-inch scopes."),
    PopularEntry("NGC253", "Sculptor Galaxy", "intermediate", "autumn",
                 "Bright, dust-laced edge-on galaxy of the southern sky."),
    PopularEntry("M8", "Lagoon Nebula", "beginner", "summer",
                 "Bright summer emission nebula near the galactic centre."),
)


def get_popular(catalog_id: str) -> PopularEntry | None:
    """Look up a popular entry by canonical catalog identifier."""
    needle = catalog_id.strip().upper().replace(" ", "")
    for entry in POPULAR:
        if entry.catalog_id.upper() == needle:
            return entry
    return None
