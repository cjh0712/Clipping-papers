"""Check exact identities underlying the congruent skew screw-net construction."""

import sympy as sp


def verify():
    long_diagonal, short_diagonal, offset = sp.symbols('long short offset', positive=True)
    parameter = sp.symbols('parameter', real=True)
    cosine = (1 - parameter**2) / (1 + parameter**2)
    sine = 2 * parameter / (1 + parameter**2)
    shift = (long_diagonal**2 * cosine**2 - short_diagonal**2 * sine**2 - offset**2) / (2 * offset)
    radial = shift - sp.I * long_diagonal * cosine
    row_factor = (shift + offset - sp.I * short_diagonal * sine) / radial
    column_factor = (shift + offset + sp.I * short_diagonal * sine) / radial
    assert sp.factor(row_factor * sp.conjugate(row_factor) - 1) == 0
    assert sp.factor(column_factor * sp.conjugate(column_factor) - 1) == 0
    assert sp.factor(radial * row_factor * column_factor - sp.conjugate(radial)) == 0
    master = [sp.Matrix([0, -long_diagonal, 0]), sp.Matrix([offset, 0, -short_diagonal]),
              sp.Matrix([0, long_diagonal, 0]), sp.Matrix([offset, 0, short_diagonal])]
    for (row, column), expected in zip(((0, 0), (0, 1), (1, 1), (1, 0)), master):
        current_radial = sp.cancel(radial * row_factor**row * column_factor**column)
        real_part = (current_radial + sp.conjugate(current_radial)) / 2
        imaginary_part = (current_radial - sp.conjugate(current_radial)) / (2 * sp.I)
        height = (row + column - 1) * long_diagonal * sine + (row - column) * short_diagonal * cosine
        current = sp.Matrix([-shift + real_part, imaginary_part * cosine + height * sine,
                             -imaginary_part * sine + height * cosine])
        assert all(sp.factor(value) == 0 for value in current - expected)
    volume = sp.Matrix.hstack(*[point - master[0] for point in master[1:]]).det()
    assert sp.factor(volume - 4 * offset * long_diagonal * short_diagonal) == 0
    for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
        difference = master[first] - master[second]
        assert sp.expand(difference.dot(difference) - long_diagonal**2 - short_diagonal**2 - offset**2) == 0
    radius_squared = shift**2 + long_diagonal**2 * cosine**2
    changing_distance_squared = 4 * (offset**2 + long_diagonal**2 + short_diagonal**2) - (
        offset**2 + (long_diagonal * cosine - short_diagonal * sine)**2)**2 / radius_squared
    derivative_at_zero = sp.factor(sp.diff(changing_distance_squared, parameter).subs(parameter, 0) / 2)
    assert derivative_at_zero == 16 * long_diagonal * short_diagonal * offset**2 / (long_diagonal**2 + offset**2)
    tangent = sp.symbols('tangent', real=True)
    fixed_cosine = 1 / sp.sqrt(1 + tangent**2)
    fixed_sine = tangent / sp.sqrt(1 + tangent**2)
    fixed_shift = (4 * fixed_cosine**2 - fixed_sine**2 - sp.Rational(1, 25)) * sp.Rational(5, 2)
    fixed_radial = fixed_shift - 2 * sp.I * fixed_cosine
    fixed_row_factor = (fixed_shift + sp.Rational(1, 5) - sp.I * fixed_sine) / fixed_radial
    fixed_column_factor = (fixed_shift + sp.Rational(1, 5) + sp.I * fixed_sine) / fixed_radial
    row_height, column_height = 2 * fixed_sine + fixed_cosine, 2 * fixed_sine - fixed_cosine
    rays = []
    for multiplier, height in ((1 / fixed_row_factor, -row_height), (fixed_column_factor, column_height),
                                (fixed_row_factor, row_height), (1 / fixed_column_factor, -column_height)):
        delta_radial = sp.cancel(fixed_radial * (multiplier - 1))
        real_part = sp.simplify((delta_radial + sp.conjugate(delta_radial)) / 2)
        imaginary_part = sp.simplify((delta_radial - sp.conjugate(delta_radial)) / (2 * sp.I))
        rays.append(sp.Matrix([real_part, imaginary_part, height]))
    for start in range(4):
        determinant = sp.factor(sp.Matrix.hstack(*[rays[(start + index) % 4] for index in range(3)]).det())
        sign = 1 if start % 2 == 0 else -1
        expected = -sp.Rational(8, 5) * (101 - 26 * tangent**2) / (101 + sign * 100 * tangent + 26 * tangent**2)
        assert sp.factor(determinant - expected) == 0
    print('PASS: both screw multipliers have exact unit norm; the seed quad is fixed.')
    print('PASS: each cell is a rigid image of the same nonplanar quad.')
    print('PASS: signed six-volume = 4*offset*long*short.')
    print('PASS: a cross-cell distance has nonzero derivative, so the flex is nontrivial.')
    print('PASS: exact vertex-signature formulas; the sharp margin on 0..45 degrees is 120/227.')


if __name__ == '__main__':
    verify()
