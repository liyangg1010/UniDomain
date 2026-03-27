#!/usr/bin/env python3
"""Quick test to verify HD extensions are properly installed and importable."""

def test_imports():
    """Test that all HD extension functions can be imported."""
    print("Testing HD Extensions imports...")

    try:
        from unidomain import (
            # Atomic Domain
            atomic_domain_pipeline_for_hd,
            initiate_domain_from_keyframes_for_hd,
            run_initial_domain_step_for_hd,
            # Domain Fusion
            domain_fusion_pipeline_hd,
            delete_type,
            merge_predicates_hd,
            merge_operators_hd,
            fuse_two_domains_hd,
            run_domain_fusion_hd,
            # LLM Utilities
            exclude_thinking,
            HD_MODEL_COSTS,
            get_extended_model_costs,
        )
        print("✅ All HD extension functions imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_exclude_thinking():
    """Test the exclude_thinking function."""
    print("\nTesting exclude_thinking function...")

    from unidomain import exclude_thinking

    test_cases = [
        ("<think>reasoning</think>Answer: 42", "Answer: 42"),
        ("No thinking tags", "No thinking tags"),
        ("<THINK>case insensitive</THINK>Result", "Result"),
    ]

    all_passed = True
    for input_str, expected in test_cases:
        result = exclude_thinking(input_str)
        if result == expected:
            print(f"  ✅ {repr(input_str[:30])}... -> {repr(result)}")
        else:
            print(f"  ❌ Expected {repr(expected)}, got {repr(result)}")
            all_passed = False

    return all_passed


def test_delete_type():
    """Test the delete_type function."""
    print("\nTesting delete_type function...")

    from unidomain import delete_type

    predicates = {
        "on(?x - block, ?y - block)": "x is on y",
        "clear(?x - block)": "x is clear"
    }

    operators = {
        "move": "parameters: ?b - block, ?from - location, ?to - location\n..."
    }

    new_preds, new_ops = delete_type(predicates, operators)

    # Check predicates
    expected_pred_key = "on(?x, ?y)"
    if expected_pred_key in new_preds:
        print(f"  ✅ Predicate type removed: {expected_pred_key}")
    else:
        print(f"  ❌ Expected predicate key not found: {expected_pred_key}")
        return False

    # Check operators
    if "- block" not in new_ops["move"]:
        print(f"  ✅ Operator type removed")
    else:
        print(f"  ❌ Operator still contains type annotation")
        return False

    return True


def test_model_costs():
    """Test HD model costs."""
    print("\nTesting HD model costs...")

    from unidomain import HD_MODEL_COSTS, get_extended_model_costs

    if "gpt-5" in HD_MODEL_COSTS:
        print(f"  ✅ HD_MODEL_COSTS contains gpt-5: {HD_MODEL_COSTS['gpt-5']}")
    else:
        print("  ❌ gpt-5 not found in HD_MODEL_COSTS")
        return False

    extended = get_extended_model_costs()
    if "gpt-4o" in extended and "gpt-5" in extended:
        print(f"  ✅ Extended costs contain both default and HD models")
        print(f"     Total models: {len(extended)}")
    else:
        print("  ❌ Extended costs incomplete")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("UniDomain HD Extensions - Installation Test")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("exclude_thinking", test_exclude_thinking),
        ("delete_type", test_delete_type),
        ("Model Costs", test_model_costs),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' raised exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All tests passed! HD extensions are ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
