from math import factorial,exp,pi,sin

# Calculate e^x using Taylor series expansion
x = 2
sum_val = 0.0
for n in range(0, 8):
    term = x**n / factorial(n) # the n-th term in the series
    sum_val += term

print(f"after {n} terms: {sum_val}")
print("the true value of e^2:", exp(2))

# Calculate sin(x) using Taylor series expansion
def my_sin(x): 
    sign = -1 
    p = d = 1 
    i = sinx = 0 
 
    while d > 0.00001:  
        d = (x**p)/float(factorial(p)) 
        sinx += ((sign**i)*d) 
        i+=1 
        p+=2 
 
    return sinx 
 
test_list = [0, pi/2, pi, 1, 0.7] 
 
print ("inbuilt sinx\tdefined sinx" )
for t in test_list: 
    print ("%.8f\t%.8f"%(sin(t),my_sin(t)))