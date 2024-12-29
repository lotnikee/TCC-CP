from copy import deepcopy
from ase.build import add_adsorbate, fcc111, molecule 
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.visualize import view
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
view(CH4_ads)

# A) Calculate adsoroption energy
adsorption_energy_CH4 = energy_CH4_ads - (energy_slab + energy_CH4_gas)
print(f"Adsorption energy of CH4 on Cu(111): {adsorption_energy_CH4: .3f} eV")

