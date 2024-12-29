from copy import deepcopy
from ase.build import add_adsorbate, fcc111, molecule 
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.thermochemistry import IdealGasThermo, HarmonicThermo

# Get the energy of a CH4 molecule 
CH4_molec = molecule("CH4")
CH4_molec.calc = EMT()
energy_CH4_gas = CH4_molec.get_potential_energy()

# Create a Cu(111) slab and get energy
slab = fcc111("Cu", size=(4,4,2), vacuum=10.0)
slab.calc = EMT()
energy_slab = slab.get_potential_energy()

# Run geometry optimisation of CO on Cu(111) slab and print adsorption energy
CH4_ads = deepcopy(slab)
add_adsorbate(slab=CH4_ads, adsorbate=CH4_molec, height=4.0, position=(3.82, 2.21))
constraint = FixAtoms(mask=[atom.symbol == "Cu" for atom in CH4_ads])
CH4_ads.set_constraint(constraint)
dyn = QuasiNewton(CH4_ads, trajectory="CH4_Cu(111).traj")
dyn.run(fmax=0.05)
energy_CH4_ads = CH4_ads.get_potential_energy()

# A) Calculate adsoroption energy
adsorption_energy_CH4 = energy_CH4_ads - (energy_slab + energy_CH4_gas)
print(f"Adsorption energy of CH4 on Cu(111): {adsorption_energy_CH4: .3f} eV")

# B) Calculate Gibb's Free energy of adsorption of CH4 on Cu(111) at 300K and 1 bar
vib_energy_CH4_gas = [0.3843, 0.3840, 0.3840, 0.3685, 0.1881, 0.1879, 0.1595, 0.1593, 0.1592]
vib_energy_CH4_ads = [0.3815, 0.3758, 0.3758, 0.3625, 0.1850, 0.1848, 0.1589, 0.1584, 0.1559, 0.0161, 0.0161, 0.0112, 0.0061, 0.0061, 0.0061]
thermo_CH4_gas = IdealGasThermo(vib_energies=vib_energy_CH4_gas,
                        geometry="nonlinear",
                        potentialenergy=energy_CH4_gas,
                        atoms=CH4_molec,
                        symmetrynumber=12,
                        spin=0)
thermo_CH4_ads = HarmonicThermo(vib_energies=vib_energy_CH4_ads, potentialenergy=energy_CH4_ads)
temp = 300
pressure = 1.0e+5

g_CH4_gas = thermo_CH4_gas.get_gibbs_energy(temperature=temp, pressure=pressure, verbose=False)
g_CH4_ads = thermo_CH4_ads.get_helmholtz_energy(temperature=temp, verbose=False)
g_slab = energy_slab
Pa_to_bar = 1.0e-5
adsorption_free_energy_CH4 = g_CH4_ads - (g_slab + g_CH4_gas)
print(f"Adsorption free energy of CH4 on Cu(111) at {temp}K and {pressure*Pa_to_bar} bar: {adsorption_free_energy_CH4: .3f} eV")

# C) Calculting the selectivity of CO over CH4
# Include calculations for CO first

# Get energy of CO molecule 
CO_molec = molecule("CO")
CO_molec.calc = EMT()
energy_CO_gas = CO_molec.get_potential_energy()

# Run geometry optimisation of CO on Cu(111) slab and print adsorption energy
CO_ads = deepcopy(slab)
add_adsorbate(slab=CO_ads, adsorbate=CO_molec, height=3.0, position=(3.82, 2.21))
constraint_CO = FixAtoms(mask=[atom.symbol == "Cu" for atom in CO_ads])
CO_ads.set_constraint(constraint_CO)
dyn_CO = QuasiNewton(CO_ads, trajectory="CO_Cu(111).traj")
dyn.run(fmax=0.05)
energy_CO_ads = CO_ads.get_potential_energy()

# Calculate adsoroption energy CO
adsorption_energy_CO = energy_CO_ads - (energy_slab + energy_CO_gas)
print(f"Adsorption energy of CO on Cu(111): {adsorption_energy_CO: .3f} eV")

# Calculate Gibb's Free energy of adsorption of CO on Cu(111) at 300K and 1 bar
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
g_slab_CO = energy_slab
Pa_to_bar = 1.0e-5
adsorption_free_energy_CO = g_CO_ads - (g_slab_CO + g_CO_gas)
print(f"Adsorption free energy of CO on Cu(111) at {temp}K and {pressure*Pa_to_bar} bar: {adsorption_free_energy_CO: .3f} eV")