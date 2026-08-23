import math

import pytest

from quant_pipeline.core.artifacts import canonical_json


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError):
        canonical_json({"value": value})
