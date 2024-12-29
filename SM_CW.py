from copy import deepcopy
from ase.build import add_adsorbate, fcc111, molecule 
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.visualize import view
from ase.thermochemistry import IdealGasThermo, HarmonicThermo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter

# Get energy of CO molecule 
CO_molec = molecule("CO")
CO_molec.calc = EMT()
energy_CO_gas = CO_molec.get_potential_energy()

# Create a Cu(111) slab and get energy
slab = fcc111("Cu", size=(4,4,2), vacuum=10.0)
slab.calc = EMT()
energy_slab = slab.get_potential_energy()

# Run geometry optimisation of CO on Cu(111) slab and print adsorption energy
CO_ads = deepcopy(slab)
add_adsorbate(slab=CO_ads, adsorbate=CO_molec, height=3.0, position=(3.82, 2.21))
constraint = FixAtoms(mask=[atom.symbol == "Cu" for atom in CO_ads])
CO_ads.set_constraint(constraint)
dyn = QuasiNewton(CO_ads, trajectory="CO_Cu(111).traj")
dyn.run(fmax=0.05)
energy_CO_ads = CO_ads.get_potential_energy()

# A) Calculate adsoroption energy
adsorption_energy_CO = energy_CO_ads - (energy_slab + energy_CO_gas)
print(f"Adsorption energy of CO on Cu(111): {adsorption_energy_CO: .3f} eV")

# B) Calculate Gibb's Free energy of adsorption of CO on Cu(111) at 300K and 1 bar
vib_energy_CO_gas = [0.2634]
vib_energy_CO_ads = [0.2404, 0.0827, 0.0601, 0.0600, 0.0072, 0.0065]
thermo_CO_gas = IdealGasThermo(vib_energies=vib_energy_CO_gas,
                        geometry="linear",
                        potentialenergy=energy_CO_gas,
                        atoms=CO_molec,
                        symmetrynumber=1,
                        spin=0)
thermo_CO_ads = HarmonicThermo(vib_energies=vib_energy_CO_ads, potentialenergy=energy_CO_ads)
temp = 300
pressure = 1.0e+5

g_CO_gas = thermo_CO_gas.get_gibbs_energy(temperature=temp, pressure=pressure, verbose=False)
g_CO_ads = thermo_CO_ads.get_helmholtz_energy(temperature=temp, verbose=False)
g_slab = energy_slab
Pa_to_bar = 1.0e-5
adsorption_free_energy_CO = g_CO_ads - (g_slab + g_CO_gas)
print(f"Adsorption free energy of CO on Cu(111) at {temp}K and {pressure*Pa_to_bar} bar: {adsorption_free_energy_CO: .3f} eV")

# C) Plot Gibb's Free energy of adsorption as a function of T and P 
x = np.linspace(100, 1000, 100)
y = np.linspace(-3, 2, 300)
X, Y = np.meshgrid(x,y)

z_min, z_max, z_interval = -0.22, 1.76, 0.22
z_values = np.arange(z_min, z_max + z_interval, z_interval)
Z = np.interp(np.linspace(0, len(z_values)-1, X.size), np.arange(len(z_values)), z_values).reshape(X.shape)
              

plt.contourf(X,Y,Z, levels=np.arange(z_min, z_max + z_interval, z_interval), cmap="viridis")
plt.colorbar(label="Adsorption Free Enegy (eV)")
plt.yscale("log")
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0))
plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.1, 10.0)*0.1, numticks=10))
plt.gca().yaxis.set_major_formatter(LogFormatter(base=10.0))
plt.title("Filled Contour Plot")
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (bar)")
plt.show()
