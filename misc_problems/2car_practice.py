import numpy as np

# Given 2 cars with r1 and r2, 
P1 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.5])
r1 = 1.5

P2 = np.array([5.0, 2.0])
V2 = np.array([-0.5, -1.0])
r2 = 1.5

# Detecting Point of collision, if any
V = V2 - V1
D0 = P2 - P1

a = np.dot(V, V)
b = 2 * np.dot(V, D0)
c = np.dot(D0, D0) - (r2 + r1)**2

temp = b**2 - 4*a*c
if temp < 0:
    print("No Solutions")
elif temp == 0:
    print("One Solution")
else:
    print("Two Solutions")
    
# Getting the actual collision points
if temp >= 0:
    t1 = (-b + np.sqrt(temp)) / 2 / a
    t2 = (-b - np.sqrt(temp)) / 2 / a
    
print(f"Collision Times {t1} {t2}")
print(f"Collision Positions Car 1")
print(P1 + V1 * t1)
print(P1 + V1 * t2)
