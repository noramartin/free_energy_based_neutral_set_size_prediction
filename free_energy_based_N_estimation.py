from functions.N_estimates import *
import functions.rna_structural_functions as cg
import sys

###############################################################################################
# print help
###############################################################################################
if str(sys.argv[1]) in ['--h', '-help']:
	print 'This program implements the free-energy based neutral set size estimation method by N.S. Martin and S.E. Ahnert (2021).\n' +\
	       'The following parameters should be given:\n1) Structure in dot-bracket notation.\n'+\
	       '2) Should a linear correction be applied (details in Supplementary Material). "0" means that no correction is applied and "1" means the opposite. Note that the correction can only be applied for a folding model with no isolated base pairs and for mfe set size predictions only.\n' +\
	       '3) Mfe set size predictions or low-energy set size predictions? ["mfe" or "low-energy"]\n' +\
	       '4) For low-energy predictions only: please give the low-energy cut-off (in kcal/modl). This should be a negative value or zero.\n\n' +\
	       'example for mfe set size prediction with no correction: python2 free_energy_based_N_estimation.py "(((...))).." 0 "mfe"\n' +\
	       'example for low-energy set size prediction with a cut-off x = -2 kcal/mol: python2 free_energy_based_N_estimation.py "(((...))).." 0 "low-energy" -2'
	sys.exit()
elif len(sys.argv) < 4:
	print 'Insufficient number of parameters. Please run with option -help for usage instructions.'
	sys.exit()
###############################################################################################
# get input
###############################################################################################
structure = str(sys.argv[1])
linear_fit = int(sys.argv[2])
type_prediction = str(sys.argv[3])

###############################################################################################
# print input
###############################################################################################
print 'input structure: ', structure
assert linear_fit in [0, 1]
assert type_prediction in ['mfe', 'low-energy', 'lowenergy']
if linear_fit == 1:
   print 'setting for linear correction: apply correction - please note that this has only been tested for a folding model with no isolated base pairs.'
elif linear_fit == 0:
   print 'setting for linear correction: NOT applying correction'
if type_prediction == 'mfe':
	print 'predict mfe set size (usually referred to simply as neutral set size)'
elif type_prediction.startswith('low'):
	print 'predict low-energy set size'
###############################################################################################
# check input
###############################################################################################
if not cg.is_likely_to_be_valid_structure(structure):
	print 'error: this structure does not meet the requirements of a valid structure (hairpin loops have at least three nucleotides, all brackets are closed and there is at least one base pair)'
if linear_fit == 1 and cg.has_length_one_stack(structure):
   print 'error: linear correction does not exist for a folding model that includes isolated base pairs - please change the input parameters'
   sys.exit()	
if linear_fit == 1 and type_prediction != 'mfe':
   print 'error: linear correction does not exist for low-energy set size calculations - please change the input parameters'
   sys.exit()
if type_prediction.startswith('low'):
	if len(sys.argv) < 5:
	   print 'Need to specify low-energy cut-off as an additional parameter. Please run with option -help for further instructions.'
	   sys.exit()
	cutoff = float(sys.argv[4])
	if cutoff > 0.001:
	   print 'Parameter error: please choose a low-energy cut-off less or equal than 0 kcal/mol.'
	   sys.exit()
	print ' set low-energy cut-off to x=', round(cutoff, 2), 'kcal/mol'
###############################################################################################
# perform computations and print
###############################################################################################
if type_prediction == 'mfe':
   N_estimate_raw = free_energy_estimate_with_heuristic(structure)
   if linear_fit == 1:
   	  L = len(structure)
   	  slope = 0.65 - 0.0023 * L
   	  intercept = - 7 + 0.64 * L
   	  print 'mfe set size estimate', np.exp(np.log(N_estimate_raw) * slope + intercept)
   else:
   	  print 'mfe set size estimate', N_estimate_raw
else:
   print 'low-energy set size estimate', low_energy_set_size_estimate(structure, G_max= cutoff) 


