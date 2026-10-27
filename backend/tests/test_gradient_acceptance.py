"""Acceptance checks for the previously misstated micrograd gradient behavior."""
import types
from pathlib import Path

import pytest


def test_chinese_gradient_question_reference_behavior():
    """The incoming output gradient is not the sum of input gradients.

    This checks the actual local micrograd implementation, not LLM output.
    """
    source = Path(__file__).resolve().parents[1] / "tmp/reposage/micrograd/micrograd/engine.py"
    if not source.is_file():
        pytest.skip("Local micrograd checkout is not available")
    module = types.ModuleType("micrograd_acceptance_engine")
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)

    a, b = module.Value(2.0), module.Value(3.0)
    result = a + b
    result.backward()
    assert (result.grad, a.grad, b.grad) == (1.0, 1.0, 1.0)
    assert result.grad != a.grad + b.grad

    x, y = module.Value(2.0), module.Value(3.0)
    product = x * y
    product.backward()
    assert (product.grad, x.grad, y.grad) == (1.0, 3.0, 2.0)
    assert product.grad != x.grad + y.grad
