"""Pure-Python prediction model (no external libraries needed).

For each transportation type we fit a linear regression:
    minutes = intercept + a * distance_km + b * weight_kg
using the normal equations, solved with Gaussian elimination.

This class lives in its own file so Pickle can load it later
(both train_model.py and server.py import it from here).
"""


def _solve(matrix, vector):
    """Solve matrix * x = vector using Gaussian elimination."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            raise ValueError("Not enough varied data to fit this transport type.")
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


class DeliveryTimeModel:
    def __init__(self):
        self.coefficients = {}  # transport -> [intercept, per_km, per_kg]

    def fit(self, records):
        """records: list of (distance_km, weight_kg, transportation, minutes)."""
        groups = {}
        for d, w, t, m in records:
            groups.setdefault(t, []).append((d, w, m))

        for transport, rows in groups.items():
            # Normal equations: (X^T X) beta = X^T y, with X = [1, distance, weight]
            xtx = [[0.0] * 3 for _ in range(3)]
            xty = [0.0] * 3
            for d, w, m in rows:
                x = [1.0, d, w]
                for i in range(3):
                    xty[i] += x[i] * m
                    for j in range(3):
                        xtx[i][j] += x[i] * x[j]
            self.coefficients[transport] = _solve(xtx, xty)
        return self

    def predict(self, distance_km, weight_kg, transportation):
        if transportation not in self.coefficients:
            raise ValueError("Unknown transportation type.")
        b0, b1, b2 = self.coefficients[transportation]
        return max(1.0, b0 + b1 * distance_km + b2 * weight_kg)

    @property
    def transport_types(self):
        return list(self.coefficients)
