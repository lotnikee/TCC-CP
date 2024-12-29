from copy import deepcopy
from ase.build import add_adsorbate, fcc111, molecule 
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.visualize import view

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
view(CO_ads)

# A) Calculate adsoroption energy
adsorption_energy_CO = energy_CO_ads - (energy_slab + energy_CO_gas)
print(f"Adsorption energy of CO on Cu(111): {adsorption_energy_CO: .3f} eV")