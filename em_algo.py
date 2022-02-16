import numpy as np
import random
import math

def normal_density(m1, v1, x1):
    if(abs(v1) < 0.000001):
        return 0
    return math.exp(- ((x1 - m1)**2) / (2*v1)) / math.sqrt(2 * math.pi)

    
def random_choose(alpha):
    z = random.uniform(0, 1)
    sum = 0
    j = 0
    while(sum <= z):
        sum += alpha[j]
        j += 1
    return j - 1

def initialize (J):
    alpha_zero = [1/J] * J
    mu_zero = [j for j in range(J)]
    v_zero = [1] * J
    return alpha_zero, mu_zero, v_zero
    

def generate_data(alpha, mu, v):
   n = 100
   X = np.zeros(n)
   for i in range(n):
       j = random_choose(alpha)
       X[i] = random.normalvariate(mu[j], v[j])
   return X

       
def transform(X, alpha_estimated, mu_estimated, v_estimated):
    n = len(X)
    J = len(alpha_estimated)
    H = np.zeros((n, J))
    for i in range(n):
        sum_row = 0
        for j in range(J):
            sum_row += alpha_estimated[j] * normal_density(mu_estimated[j], v_estimated[j], X[i])
        for j in range(J):
            H[i][j] = alpha_estimated[j] * normal_density(mu_estimated[j], v_estimated[j], X[i]) / sum_row
        
    for j in range(J):
        sum_column = 0
        sum_median = 0
        for i in range(n):
            sum_median += X[i] * H[i][j]
            sum_column += H[i][j]
        
        alpha_estimated[j] = sum_column / n
        mu_estimated[j] = sum_median / sum_column
        
        sum_variance = 0
        for i in range(n):
            sum_variance += (X[i] - mu_estimated[j])**2 * H[i][j]
        v_estimated[j] = sum_variance / sum_column
            
    return alpha_estimated, mu_estimated, v_estimated

alpha = [0.3, 0.3, 0.4]
mu = [10, 20, -10]
v = [1, 1, 1]

X = generate_data(alpha, mu, v)

alpha_estimated, mu_estimated, v_estimated = [0.1, 0.1, 0.8], [9, 15, -4], [4, 4, 4]

number_of_iterations = 1000
for iteration in range(number_of_iterations):
    alpha_esimated, mu_estimated, v_estimated = transform(X, alpha_estimated, mu_estimated, v_estimated)
print(alpha_estimated)
print(mu_estimated)
print(v_estimated)
            