import numpy as np
import matplotlib.pyplot as plt
from mittag_leffler import mittag_leffler  
from latexify import latexify  

# Apply LaTeX-style figure settings
latexify(columns=2)  # Adjust columns or dimensions as needed

# Parameters for the function
alpha = 1.0
beta = 1.0
z = np.linspace(0, 5, 500)

# Values of num_terms to plot
num_terms_values = [2, 5, 10, 100]

for num_terms in num_terms_values:
    # Compute mittag-leffler values
    mittag_values = np.array([mittag_leffler(alpha, beta, zi, num_terms) for zi in z])
    
    # Create a single plot for each num_terms
    plt.figure()
    
    # Plot numerical graph
    plt.plot(z, mittag_values, label=r"Series", color="blue", linestyle="--")
    
    # Plot analytical graph with markers only
    nth = 20  # Plot markers every 20th point
    plt.plot(z[::nth], np.exp(z[::nth]), label=r"Closed form", color="red", marker="o", markersize=2, linestyle="None")
    
    # Add zero lines
    plt.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.axvline(0, color="black", linewidth=0.8, linestyle=":")
    
    # Titles and labels
    #plt.title(rf"Mittag-Leffler Function ($\mathrm{{num\_terms}} = {num_terms}$)")
    plt.xlabel("z")
    plt.ylabel(r"$E_{1,1}(z)$")
    plt.legend()
    plt.grid()
    
    # Save each figure as a separate PDF
    filename = f"ml_terms_{num_terms}.pdf"
    plt.savefig(filename, format="pdf", transparent=True, bbox_inches="tight", pad_inches=0.0)
    plt.close()  # Close the figure to avoid overlap
