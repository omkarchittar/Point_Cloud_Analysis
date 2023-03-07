import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


print("*************************************************************************")
print("Problem 2.1): Calculate the covariance matrix: ")
df = pd.read_csv(r'pc1.csv', header=None)

x1 = df[df.columns[0]].to_numpy()
y1 = df[df.columns[1]].to_numpy()
z1 = df[df.columns[2]].to_numpy()

df = pd.read_csv(r'pc2.csv', header=None)

x2 = df[df.columns[0]].to_numpy()
y2 = df[df.columns[1]].to_numpy()
z2 = df[df.columns[2]].to_numpy()

n = len(x1)
mean_x1 = sum(x1) / n
mean_y1 = sum(y1) / n
mean_z1 = sum(z1) / n

cov_matrix = np.zeros((3, 3))

for i in range(0,300):
    cov_matrix[0, 0] += (x1[i] - mean_x1)**2
    cov_matrix[1, 1] += (y1[i] - mean_y1)**2
    cov_matrix[2, 2] += (z1[i] - mean_z1)**2
    cov_matrix[0, 1] += (x1[i] - mean_x1) * (y1[i] - mean_y1)
    cov_matrix[0, 2] += (x1[i] - mean_x1) * (z1[i] - mean_z1)
    cov_matrix[1, 2] += (y1[i] - mean_y1) * (z1[i] - mean_z1)

cov_matrix[0, 1] /= n
cov_matrix[0, 2] /= n
cov_matrix[1, 2] /= n
cov_matrix[1, 0] = cov_matrix[0, 1]
cov_matrix[2, 0] = cov_matrix[0, 2]
cov_matrix[2, 1] = cov_matrix[1, 2]

cov_matrix[0, 0] /= n
cov_matrix[1, 1] /= n
cov_matrix[2, 2] /= n

print("\nThe Covariance Matrix is:")
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues of the covariance matrix are:",eigenvalues)

print("\nThe surface normal is the Eigenvector corresponding to the smallest Eigenvalue")
print("Therefore,")

magnitude = np.sqrt(eigenvalues[0])
direction = eigenvectors[:, np.argmin(eigenvalues)]

a, b, c = direction

print("Direction of the surface normal:", direction)
print("Magnitude of the surface normal:",magnitude)
print("\n*************************************************************************")

fig1 = plt.figure(figsize=(12, 12))
ax = fig1.add_subplot(projection='3d')
ax.scatter(x1, y1, z1, color= 'blue')
ax.quiver(a, b, c, 2, 2, 2, length = 2*magnitude, color = 'red', alpha = 1)


################################################################################################
points1 = np.column_stack((x1,y1,z1))
points2 = np.column_stack((x2,y2,z2))
################################################################################################

print("\nProblem 2.2): Estaimation using Standard Least Squares, Total Least Squares and RANSAC ")

# Least square fitting pc1
A = np.column_stack((x1, y1, np.ones_like(x1)))

# Define the vector b
b = np.row_stack(z1)

# Calculate the normal equations
ATA = np.dot(A.T, A)
ATAInv = np.linalg.inv(ATA)
ATb = np.dot(A.T, b)
x = np.dot(ATAInv, ATb)

a, b, c = x

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
eq = a * x + b * y + c

fig2 = plt.figure(figsize=(12, 12))
fig2.suptitle('LEAST SQUARE fitting pc1', fontsize=16)
bx = fig2.add_subplot(projection='3d')
bx.plot_surface(x, y, eq, color = 'blue', alpha = 0.4)
bx.scatter(x1, y1, z1, color = 'red', marker = 'o')
# plt.show()

################################################################################################

# LS pc2
A = np.column_stack((x2, y2, np.ones_like(x2)))

# Define the vector b
b = np.row_stack(z2)

# Calculate the normal equations
ATA = np.dot(A.T, A)
ATAInv = np.linalg.inv(ATA)
ATb = np.dot(A.T, b)

x = np.dot(ATAInv, ATb)

# Extract the optimized coefficients
a, b, c = x

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
eq = a * x + b * y + c

fig3 = plt.figure(figsize=(12, 12))
fig3.suptitle('LEAST SQUARE fitting pc2', fontsize=16)
cx = fig3.add_subplot(projection='3d')
cx.plot_surface(x, y, eq, color = 'blue', alpha = 0.4)
cx.scatter(x2, y2, z2, color = 'red', marker = 'o')
# plt.show()

################################################################################################
# TLS pc1
A = np.column_stack((x1, y1, z1))
A_mean = np.mean(A, axis=0)
A_centered = A - A_mean

# Calculate the covariance matrix of the centeblue data matrix
cov = np.dot(A_centered.T, A_centered)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigvals, eigvecs = np.linalg.eig(cov)

# Extract the eigenvector corresponding to the smallest eigenvalue
n = eigvecs[:, np.argmin(eigvals)]

# Calculate the optimized coefficients
a, b, c = n[:3]
d = -(a*A_mean[0] + b*A_mean[1] + c*A_mean[2])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
eq = (-a/c) * x + (-b/c) * y + (-d/c)

fig4 = plt.figure(figsize=(12, 12))
fig4.suptitle('TOTAL LEAST SQUARE fitting pc1', fontsize=16)
dx = fig4.add_subplot(projection='3d')
dx.plot_surface(x, y, eq, color = 'blue', alpha = 0.4)
dx.scatter(x1, y1, z1, color = 'red', marker = 'o')


################################################################################################

# TLS pc2
A = np.column_stack((x2, y2, z2))
A_mean = np.mean(A, axis=0)
A_centered = A - A_mean

# Calculate the covariance matrix of the centered data matrix
cov = np.dot(A_centered.T, A_centered)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigvals, eigvecs = np.linalg.eig(cov)

# Extract the eigenvector corresponding to the smallest eigenvalue
n = eigvecs[:, np.argmin(eigvals)]

# Calculate the optimized coefficients
a, b, c = n[:3]
d = -(a*A_mean[0] + b*A_mean[1] + c*A_mean[2])

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x, y = np.meshgrid(x, y)
eq = (-a/c) * x + (-b/c) * y + (-d/c)

fig5 = plt.figure(figsize=(12, 12))
fig5.suptitle('TOTAL LEAST SQUARE fitting pc2', fontsize=16)
ex = fig5.add_subplot(projection='3d')
ex.plot_surface(x, y, eq, color = 'blue', alpha = 0.4)
ex.scatter(x2, y2, z2, color = 'red', marker = 'o')

################################################################################################

# RANSAC Function
def fit_surface(points, max_iterations=1000, threshold=0.1):
    best_model = None
    best_inliers = None
    best_inlier_count = 0
    num_points = points.shape[0]
    
    for i in range(max_iterations):
        # Select 3 random indices to fit a plane
        sample_indices = np.random.randint(0, num_points, 3)
        
        # Get the corresponding points
        p1, p2, p3 = points[sample_indices]
        
        # Fit a plane using 3 points
        normal = np.cross(p2 - p1, p3 - p1)
        d = -np.dot(normal, p1)
        a, b, c = normal
        
        # Calculate the distances between the plane and all other points
        distances = np.abs(normal[0] * points[:,0] + 
                           normal[1] * points[:,1] + 
                           normal[2] * points[:,2] + d) / np.linalg.norm(normal)
        
        # Find the inliers that have a distance less than the threshold
        inliers = np.where(distances < threshold)[0]
        inlier_count = len(inliers)
        
        # If we have found a better model, update the best model
        if inlier_count > best_inlier_count:
            best_model = (a, b, c, d)
            best_inliers = inliers
            best_inlier_count = inlier_count
    
    return best_model, best_inliers

################################################################################################

# RANSAC pc1
print("\nRANSAC fitting for pc1:")
surface, inliers = fit_surface(points1)
a, b, c, d = surface
print(f"Equation of the plane is: \n{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
print("\nInliers:", inliers)

x1, y1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z1 = (-a/c) * x1 + (-b/c) * y1 + (-d/c)

# Plotting
fig6 = plt.figure(figsize=(12, 12))
fig6.suptitle('RANSAC fitting pc1', fontsize=16)
fx = fig6.add_subplot(111, projection='3d')
fx.scatter(points1[inliers,0], points1[inliers,1], points1[inliers,2], color ='blue', marker='o', alpha = 1)
fx.scatter(points1[:,0], points1[:,1], points1[:,2], color='red', marker='.', alpha = 1)
fx.plot_surface(x1, y1, z1, alpha = 0.4)

######################################################################################

# RANSAC pc2
print("\nRANSAC fitting for pc2:")
surface, inliers = fit_surface(points2)
a, b, c, d = surface
print(f"Equation of the plane is:\n{a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
print("\nInliers:", inliers)

x2, y2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
z2 = (-a/c) * x2 + (-b/c) * y2 + (-d/c)

# Plot the surface
fig7 = plt.figure(figsize=(12, 12))
fig7.suptitle('RANSAC fitting pc2', fontsize=16)
gx = fig7.add_subplot(projection='3d')
gx.scatter(points2[inliers,0], points2[inliers,1], points2[inliers,2], color ='blue', marker='o', alpha = 1)
gx.scatter(points2[:,0], points2[:,1], points2[:,2], color ='red', marker='.', alpha = 1)
gx.plot_surface(x2, y2, z2 , alpha = 0.4)
plt.show()
print("*************************************************************************")

################################################################################################