# Plot the integrand of the Green function for various values of \xi

import numpy as np
import matplotlib.pyplot as plt
from mittag_leffler import mittag_leffler  
from latexify import latexify  

# Apply LaTeX-style figure settings
latexify(columns=1)  # Adjust columns or dimensions as needed

# Parameters for the function
alpha = 1
k = np.linspace(0, 160, 500) # since the function has to be integrated over k, it has to be
# bounded for ALL values of k. Show for a range of k values, that the function is indeed bounded.

lam=0.01
t=2

xi=1.0
xi1=xi
values_num_alpha_xi1 = np.array([mittag_leffler(alpha,alpha, -lam**2 * ki**2 * t**alpha,150)*np.cos(ki*xi) for ki in k])

xi=0.1
xi2=xi
values_num_alpha_xi2 = np.array([mittag_leffler(alpha,alpha, -lam**2 * ki**2 * t**alpha,150)*np.cos(ki*xi) for ki in k])

plt.figure()

plt.plot(k, values_num_alpha_xi1, '--' ,label=r"$\xi$="+str(xi1), color="blue")
plt.plot(k, values_num_alpha_xi2 ,'-', label=r"$\xi$="+str(xi2), color="black")

# Titles and labels
plt.xlabel("k")
plt.ylabel(r"$E_{\alpha,\alpha}(-D^2 k^2 t^\alpha )\cos{(k \xi)}$")
plt.grid(True)
plt.legend()

# Save the log-log scale error plot
plt.savefig("green_function_integrand_xi.pdf", format="pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)

# Show both plots
plt.show()