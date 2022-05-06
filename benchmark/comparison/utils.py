from benchmark.comparison.methods_names import MethodsNames


def two_methods_to_key(method_a: MethodsNames, method_b: MethodsNames) -> str:
    return f'({method_a}, {method_b})'
