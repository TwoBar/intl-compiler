"""Tests for intl.router — O(1) profile → adapter lookup (issue #11)."""
import pytest
from intl.router import (
    route, list_profiles, is_trained,
    RouteResult, RouterError, ADAPTER_MAP,
)

# All 24 expected profiles from spec §5.3
ALL_PROFILES = [
    "python_fastapi", "sql_postgres",
    "python_django", "python_flask", "typescript_express", "sql_mysql",
    "typescript_nextjs", "php_laravel", "php_vanilla", "javascript_vanilla",
    "html_jinja2", "html_blade", "css_tailwind",
    "java_spring", "csharp_dotnet", "go_gin", "ruby_rails", "sql_tsql", "sql_sqlite",
    "swift_ios", "kotlin_android", "dart_flutter",
    "rust_axum", "cpp_modern",
]


# ── ADAPTER_MAP ───────────────────────────────────────────────────────────────
class TestAdapterMap:
    def test_map_has_24_entries(self):
        assert len(ADAPTER_MAP) == 24

    def test_all_profiles_present(self):
        for p in ALL_PROFILES:
            assert p in ADAPTER_MAP, f"Missing profile: {p}"

    def test_each_entry_has_required_keys(self):
        required = {"profile", "phase", "threshold", "pairs", "adapter_id", "base_model"}
        for profile, meta in ADAPTER_MAP.items():
            missing = required - set(meta.keys())
            assert not missing, f"{profile} missing keys: {missing}"

    def test_phases_are_0_to_5(self):
        phases = {m["phase"] for m in ADAPTER_MAP.values()}
        assert phases == {0, 1, 2, 3, 4, 5}

    def test_thresholds_in_valid_range(self):
        for profile, meta in ADAPTER_MAP.items():
            t = meta["threshold"]
            assert 0.0 < t <= 1.0, f"{profile} threshold {t} out of range"

    def test_pairs_positive(self):
        for profile, meta in ADAPTER_MAP.items():
            assert meta["pairs"] > 0, f"{profile} has non-positive pairs"


# ── route() ───────────────────────────────────────────────────────────────────
class TestRoute:
    def test_returns_route_result(self):
        result = route("python_fastapi")
        assert isinstance(result, RouteResult)

    def test_profile_matches(self):
        result = route("python_fastapi")
        assert result.profile == "python_fastapi"

    def test_phase_0_profiles(self):
        assert route("python_fastapi").phase == 0
        assert route("sql_postgres").phase == 0

    def test_phase_5_profiles(self):
        assert route("rust_axum").phase == 5
        assert route("cpp_modern").phase == 5

    def test_adapter_id_contains_profile(self):
        result = route("python_fastapi")
        assert "python_fastapi" in result.adapter_id

    def test_base_model_is_qwen(self):
        result = route("python_fastapi")
        assert "Qwen" in result.base_model or "qwen" in result.base_model.lower()

    def test_threshold_python(self):
        result = route("python_fastapi")
        assert result.threshold == pytest.approx(0.90)

    def test_threshold_sql(self):
        result = route("sql_postgres")
        assert result.threshold == pytest.approx(0.95)

    def test_threshold_systems(self):
        result = route("rust_axum")
        assert result.threshold == pytest.approx(0.80)
        result = route("cpp_modern")
        assert result.threshold == pytest.approx(0.80)

    def test_result_is_frozen(self):
        result = route("python_fastapi")
        with pytest.raises((AttributeError, TypeError)):
            result.profile = "modified"  # type: ignore

    @pytest.mark.parametrize("profile", ALL_PROFILES)
    def test_all_profiles_route_successfully(self, profile):
        result = route(profile)
        assert result.profile == profile

    def test_unknown_profile_raises_router_error(self):
        with pytest.raises(RouterError):
            route("nonexistent_target")

    def test_empty_profile_raises_router_error(self):
        with pytest.raises(RouterError):
            route("")

    def test_case_sensitive(self):
        with pytest.raises(RouterError):
            route("Python_FastAPI")

    def test_router_error_lists_known_profiles(self):
        with pytest.raises(RouterError) as exc_info:
            route("totally_unknown")
        assert "python_fastapi" in str(exc_info.value)

    def test_route_is_o1_no_inference(self):
        """Route must complete in <1ms (pure dict lookup, no inference)."""
        import time
        start = time.perf_counter()
        for _ in range(1000):
            route("python_fastapi")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"route() too slow: {elapsed:.3f}s for 1000 calls"


# ── list_profiles() ───────────────────────────────────────────────────────────
class TestListProfiles:
    def test_returns_all_24_without_filter(self):
        profiles = list_profiles()
        assert len(profiles) == 24

    def test_sorted(self):
        profiles = list_profiles()
        assert profiles == sorted(profiles)

    def test_phase_0(self):
        profiles = list_profiles(phase=0)
        assert set(profiles) == {"python_fastapi", "sql_postgres"}

    def test_phase_5(self):
        profiles = list_profiles(phase=5)
        assert set(profiles) == {"rust_axum", "cpp_modern"}

    def test_phase_2(self):
        profiles = list_profiles(phase=2)
        expected = {
            "typescript_nextjs", "php_laravel", "php_vanilla",
            "javascript_vanilla", "html_jinja2", "html_blade", "css_tailwind",
        }
        assert set(profiles) == expected

    def test_nonexistent_phase_returns_empty(self):
        profiles = list_profiles(phase=99)
        assert profiles == []

    def test_no_duplicates(self):
        profiles = list_profiles()
        assert len(profiles) == len(set(profiles))


# ── is_trained() ─────────────────────────────────────────────────────────────
class TestIsTrained:
    def test_known_profile_with_pairs(self):
        assert is_trained("python_fastapi") is True

    def test_all_known_profiles_have_pairs(self):
        for profile in ALL_PROFILES:
            assert is_trained(profile) is True, f"{profile} reports not trained"

    def test_unknown_profile_returns_false(self):
        assert is_trained("nonexistent_profile") is False

    def test_empty_string_returns_false(self):
        assert is_trained("") is False


# ── Phase groupings match spec ────────────────────────────────────────────────
class TestPhaseGroupings:
    def test_phase_1_profiles(self):
        profiles = set(list_profiles(phase=1))
        assert profiles == {"python_django", "python_flask", "typescript_express", "sql_mysql"}

    def test_phase_3_profiles(self):
        profiles = set(list_profiles(phase=3))
        assert profiles == {"java_spring", "csharp_dotnet", "go_gin", "ruby_rails",
                            "sql_tsql", "sql_sqlite"}

    def test_phase_4_profiles(self):
        profiles = set(list_profiles(phase=4))
        assert profiles == {"swift_ios", "kotlin_android", "dart_flutter"}

    def test_total_phases_cover_all_profiles(self):
        all_from_phases: set[str] = set()
        for ph in range(6):
            all_from_phases.update(list_profiles(phase=ph))
        assert all_from_phases == set(ALL_PROFILES)
