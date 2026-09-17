"""Unit tests for the tool capability catalog.

The primary purpose of this suite is anti-hallucination: every field name in
the catalog must correspond to a real, currently-defined field on
``ProcessingProfileConfig``, and every real field must be represented in the
catalog (nothing silently missing, nothing silently invented).
"""

from __future__ import annotations

import pytest

from app.domain.profile import ProcessingProfileConfig
from app.pipeline.adaptive.tool_catalog import (
    ADAPTIVE_STEP_FIELDS,
    TOOL_CAPABILITIES,
    ToolName,
    ValueType,
    capabilities_for_tool,
    get_capability,
    render_capabilities_for_prompt,
)

_REAL_FIELD_NAMES = frozenset(ProcessingProfileConfig.model_fields.keys())

_PYDANTIC_TO_VALUE_TYPE = {
    bool: ValueType.BOOL,
    int: ValueType.INT,
    float: ValueType.FLOAT,
    str: ValueType.STR,
}


class TestCatalogGroundedInRealProfile:
    def test_no_hallucinated_field_names(self) -> None:
        """Every catalog entry must reference a real ProcessingProfileConfig field."""
        catalog_names = {cap.field_name for cap in TOOL_CAPABILITIES}
        hallucinated = catalog_names - _REAL_FIELD_NAMES
        assert not hallucinated, f"Catalog references non-existent fields: {hallucinated}"

    def test_full_coverage_of_real_fields(self) -> None:
        """Every real ProcessingProfileConfig field must have a catalog entry."""
        catalog_names = {cap.field_name for cap in TOOL_CAPABILITIES}
        missing = _REAL_FIELD_NAMES - catalog_names
        assert not missing, f"Catalog is missing fields: {missing}"

    def test_no_duplicate_field_names(self) -> None:
        names = [cap.field_name for cap in TOOL_CAPABILITIES]
        assert len(names) == len(set(names)), "Catalog contains duplicate field_name entries"

    def test_value_type_matches_real_field_annotation(self) -> None:
        """The declared value_type must match the actual Pydantic field type."""
        mismatches = []
        for cap in TOOL_CAPABILITIES:
            real_annotation = ProcessingProfileConfig.model_fields[cap.field_name].annotation
            expected = _PYDANTIC_TO_VALUE_TYPE.get(real_annotation)
            if expected is not None and expected != cap.value_type:
                mismatches.append((cap.field_name, cap.value_type, expected))
        assert not mismatches, f"value_type mismatches (field, declared, actual): {mismatches}"

    def test_choices_are_consistent_with_default_value(self) -> None:
        """When choices are declared, the real field's default must be one of them."""
        mismatches = []
        for cap in TOOL_CAPABILITIES:
            if cap.choices is None:
                continue
            default = ProcessingProfileConfig.model_fields[cap.field_name].default
            if default not in cap.choices:
                mismatches.append((cap.field_name, default, cap.choices))
        assert not mismatches, f"Default not in declared choices: {mismatches}"


class TestGetCapability:
    def test_returns_matching_entry(self) -> None:
        cap = get_capability("denoise_strength")
        assert cap.field_name == "denoise_strength"
        assert cap.tool is ToolName.COSMIC_CLARITY

    def test_unknown_field_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            get_capability("this_field_does_not_exist")


class TestCapabilitiesForTool:
    @pytest.mark.parametrize("tool", list(ToolName))
    def test_every_tool_has_at_least_one_capability(self, tool: ToolName) -> None:
        assert len(capabilities_for_tool(tool)) > 0

    def test_filters_correctly(self) -> None:
        astap_caps = capabilities_for_tool(ToolName.ASTAP)
        assert all(cap.tool is ToolName.ASTAP for cap in astap_caps)
        assert {cap.field_name for cap in astap_caps} == {
            "plate_solving_enabled",
            "plate_solving_radius_deg",
            "plate_solving_speed",
        }


class TestRenderCapabilitiesForPrompt:
    def test_renders_one_line_per_field_in_order(self) -> None:
        text = render_capabilities_for_prompt(["stretch_strength", "denoise_strength"])
        lines = text.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("stretch_strength (siril):")
        assert lines[1].startswith("denoise_strength (cosmic_clarity):")

    def test_pipeline_owned_field_uses_pipeline_label(self) -> None:
        text = render_capabilities_for_prompt(["max_retries"])
        assert text.startswith("max_retries (pipeline):")

    def test_unknown_field_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            render_capabilities_for_prompt(["not_a_real_field"])


class TestToolCapabilityIsFrozen:
    def test_mutation_raises(self) -> None:
        cap = get_capability("denoise_strength")
        with pytest.raises(Exception):  # pydantic ValidationError on frozen model
            cap.field_name = "mutated"  # type: ignore[misc]


class TestAdaptiveStepFields:
    def test_all_referenced_fields_exist_in_catalog(self) -> None:
        for _step_name, fields in ADAPTIVE_STEP_FIELDS.items():
            for field_name in fields:
                # Raises KeyError (test failure) if hallucinated/renamed.
                get_capability(field_name)

    def test_all_referenced_fields_exist_on_real_profile(self) -> None:
        for fields in ADAPTIVE_STEP_FIELDS.values():
            for field_name in fields:
                assert field_name in _REAL_FIELD_NAMES

    def test_stretch_color_fields_match_expected_set(self) -> None:
        assert set(ADAPTIVE_STEP_FIELDS["stretch_color"]) == {
            "stretch_method",
            "stretch_strength",
            "color_calibration_enabled",
            "camera_defiltered",
            "photometric_calibration_enabled",
        }
