import json
import math

import pytest

from dendro.materials import (
    CURRENT_WALPOLE_PROFILE_VERSION,
    WALPOLE_PROFILE_ID,
    WALPOLE_PROFILE_V1,
    WalpoleMaterialProfile,
    load_walpole_profile,
)


def test_load_walpole_profile_returns_versioned_profile():
    profile = load_walpole_profile()

    assert isinstance(profile, WalpoleMaterialProfile)
    assert profile.profile_id == WALPOLE_PROFILE_ID
    assert profile.version == CURRENT_WALPOLE_PROFILE_VERSION
    assert profile.town == "Walpole"
    assert profile.state == "NH"
    assert profile.built_year_range == (1760, 1800)
    assert profile.material_group_ids() == ("hemlock", "white_pine", "hard_pine", "oak", "chestnut")
    assert profile.supported_material_groups() == ("hemlock", "white_pine", "hard_pine", "oak")
    assert profile.required_material_groups() == ("chestnut",)
    assert WALPOLE_PROFILE_V1.material_group_ids() == profile.material_group_ids()


def test_species_and_group_lookup_cover_walpole_target_materials():
    profile = load_walpole_profile()

    assert profile.material_group_for_species("TSCA") == "hemlock"
    assert profile.material_group_for_species("Eastern hemlock") == "hemlock"
    assert profile.material_group_for_species("PIST") == "white_pine"
    assert profile.material_group_for_species("white pine") == "white_pine"
    assert profile.material_group_for_species("PIRI") == "hard_pine"
    assert profile.material_group_for_species("PIPA") == "hard_pine"
    assert profile.material_group_for_species("QUAL") == "oak"
    assert profile.material_group_for_species("QUPR") == "oak"
    assert profile.material_group_for_species("QURU") == "oak"
    assert profile.material_group_for_species("CHTH") == "chestnut"
    assert profile.material_group_for_species("american chestnut") == "chestnut"
    assert profile.material_group_for_species("unknown") is None


@pytest.mark.parametrize(
    ("member_type", "expected_top_group"),
    [
        ("frame", "white_pine"),
        ("brace", "oak"),
        ("sill", "oak"),
        ("joist", "white_pine"),
        ("rafter", "white_pine"),
        ("board", "white_pine"),
        ("unknown", "white_pine"),
    ],
)
def test_member_type_priors_are_normalized_and_use_expected_materials(member_type, expected_top_group):
    profile = load_walpole_profile()

    weights = profile.member_type_prior(member_type)
    assert weights, f"missing member type prior for {member_type}"
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert max(weights, key=weights.get) == expected_top_group
    assert "chestnut" in weights


def test_context_prior_weights_are_normalized_and_chestnut_is_required():
    profile = load_walpole_profile()

    assert math.isclose(sum(profile.context_prior_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9)
    assert profile.group("chestnut").support_status == "required_coverage"
    assert profile.group("hard_pine").support_status == "supported_with_caution"


def test_profile_serializes_to_plain_json_payload():
    profile = load_walpole_profile()

    payload = profile.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert '"profile_id": "walpole_nh_late_1700s_house"' in encoded
    assert '"url": "https://www.historicnewengland.org/property/gilman-garrison-house/"' in encoded
    assert payload["citations"][0]["url"].startswith("https://")
    assert payload["material_groups"][4]["support_status"] == "required_coverage"


def test_unknown_profile_version_is_rejected():
    with pytest.raises(KeyError):
        load_walpole_profile("999.0.0")
