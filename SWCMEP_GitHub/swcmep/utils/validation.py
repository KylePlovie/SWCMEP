import math

def finite_float(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result

def positive_float(value, name: str, *, allow_zero: bool=False) -> float:
    result = finite_float(value, name)
    if result < 0 or (result == 0 and (not allow_zero)):
        condition = 'non-negative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be {condition}')
    return result

def integer_value(value, name: str, *, minimum: int=1) -> int:
    result = finite_float(value, name)
    if not result.is_integer() or result < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(result)
