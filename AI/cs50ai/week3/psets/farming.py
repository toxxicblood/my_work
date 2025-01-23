import scipy.optimize

# Objective function: 500C1 + 400C2
# Constraint 1: 3C1 + 2C2 <= 12
# Constraint 2: C1 <= 10
# Constraint 3: C2 <= 4

result = scipy.optimize.linprog(
    [-500, -400],  # Cost function: 500C1 + 400C2
    A_ub=[[3, 2]], # Coefficients for inequalities
    b_ub=[12], # Constraints for inequalities: 12, 10, 4
    bounds=[(0, 10), (0, 4)] # Bounds for C1 and C2
)

if result.success:
    print(f"X1: {round(result.x[0], 2)} hours")
    print(f"X2: {round(result.x[1], 2)} hours")
else:
    print("No solution")