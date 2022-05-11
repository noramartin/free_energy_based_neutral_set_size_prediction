import numpy as np
import rna_structural_functions as cg
from scipy.optimize import curve_fit

G_max_bp = 3


# references for the energy model
# [1] Mathews, D.H., Disney, M.D., Childs, J.L., Schroeder, S.J., Zuker, M. and Turner, D.H., 2004.
# Incorporating chemical modification constraints into a dynamic programming algorithm for prediction of RNA secondary structure. 
# Proceedings of the National Academy of Sciences, 101(19), pp.7287-7292;
# [2] Mathews, D.H., Sabina, J., Zuker, M. and Turner, D.H., 1999. 
# Expanded sequence dependence of thermodynamic parameters improves prediction of RNA secondary structure. 
# Journal of molecular biology, 288(5), pp.911-940.

###############################################################################################
# loop free energies
###############################################################################################

def loop_length_vs_G(loop_type, loop_length):
   """ return the initiation free energy for a loop of the given type and length
   for loops with several pieces (such as the two segments in the internal loop, the loop_length variable is a tuple/list)
   """
   if loop_type.startswith('internal'):
      # deltaG initiation and deltaG assymetry from table 3 in [1]
      l1, l2 = loop_length
      if l1 + l2 == 2:
         return 0.5 # highly sequence-dependent; this is average
      elif l1 + l2 == 3:
         return 2.2 # highly sequence-dependent; this is average
      elif l1 + l2 == 4:
         return 1.1  + 0.6 * abs(l1 - l2)
      elif l1 + l2 == 5:
         return 2.1 + 0.6 * abs(l1 - l2) #initiation + asymmetry 
      elif l1 + l2 == 6:
         return 1.9 + 0.6 * abs(l1 - l2) #initiation + asymmetry
      else:
         return round(1.9 + 1.08 * np.log((l1+l2)/6.0) + 0.6 * abs(l1 - l2), 1) #initiation + asymmetry
   elif loop_type.startswith('bulge'):
      # deltaG initiation from table 9 in [2]
      if loop_length == 1:
         return 3.8
      elif loop_length <= 6:
         return 2.8 + 0.4 * (loop_length - 2)
      else:
         return round(4.4 + 1.75 * 0.6163207755 * np.log(loop_length/6), 1)  #0.6163207755 is kbT in kcal/mol
   elif loop_type.startswith('hairpin'):
      # deltaG initiation and deltaG assymetry from table 1 in [1]
      if loop_length <= 9:
         return {3: 5.4, 4: 5.6, 5: 5.7, 6: 5.4, 7: 6.0, 8: 5.5, 9: 6.4}[loop_length]
      else:
         return round(6.4 + 1.75 * 0.6163207755 * np.log(loop_length/9), 1) #0.6163207755 is kbT in kcal/mol
   elif loop_type.startswith('multi'):
      # equation for deltaG multibranch initiation
      a = 9.3
      no_helices = len(loop_length) 
      return a + no_helices * (-0.9) #+ strain
   elif loop_type.startswith('exterior'):
      # no energy for exterior loop
      return 0.0
   else:
      raise RuntimeError('unknown loop type: ' + loop_type + ' of length ' + str(loop_length))

def loop_const_free_energy(structure):
   loop_types_and_len = get_roles_and_lengths_of_loops(structure)
   G_const = np.sum([loop_length_vs_G(loop_type, loop_length) for loop_type, loop_length in loop_types_and_len])
   return G_const

def get_roles_and_lengths_of_loops(structure):
   """return a list with the loop information needed for the loop energy calculations"""
   roles_in_structure = [cg.find_role_in_structure(pos, structure)[0] for pos in range(len(structure))]
   loop_types_and_len = []
   for pos, role in enumerate(roles_in_structure):
      if 'internal' in role and structure[pos-1] == '(': # only opening side of internal loop
         loop_types_and_len.append((role, cg.return_length_of_both_sides_internal_loop(pos, structure)))
      elif ('bulge' in role or 'hairpin' in role) and roles_in_structure[pos-1] != roles_in_structure[pos]:
         loop_types_and_len.append((role, cg.find_element_length(pos, structure)))
      elif 'multi' in role and roles_in_structure[pos-1] != roles_in_structure[pos]:
         branch_stem_vs_multiloop_sites_before, all_sites_in_multiloop = cg.return_complete_multiloop(pos, structure)
         if pos == min(all_sites_in_multiloop): # only count each multiloop once
            loop_types_and_len.append((role, {branch: len(ml_sites) for branch, ml_sites in branch_stem_vs_multiloop_sites_before.iteritems()}))
   if 'exterior loop' in roles_in_structure:
      pos_vs_no_bp = {pos: structure[:pos].count('(') for pos in range(len(structure) + 1) if structure[:pos].count('(') == structure[:pos].count(')')}
      nbp_vs_ext_len = {ext_index: len([pos for pos in pos_vs_no_bp if pos_vs_no_bp[pos] == nbp]) - 1 for ext_index, nbp in enumerate(sorted(list(set(pos_vs_no_bp.values()))))}
      loop_types_and_len.append(('exterior loop' ,nbp_vs_ext_len))
   return loop_types_and_len

###############################################################################################
# other characterisics
###############################################################################################


def number_of_dangling_end_terminal_mismatch_terms(structure):
   """estimate the number of dangling end/terminal mismatch terms in the structure (simplified version of the overdangle energy model)"""
   loop_types_and_len = get_roles_and_lengths_of_loops(structure)
   return np.sum([number_of_dangling_end_terminal_mismatch_in_specific_loop(loop_type, loop_length) for loop_type, loop_length in loop_types_and_len])

def number_of_dangling_end_terminal_mismatch_in_specific_loop(loop_type, loop_length):
   if 'internal' in loop_type: 
      if min(loop_length) >= 2 and max(loop_length) >= 3:
         return 2 # one terminal mismatch term at each end
      else:
         return 0
   elif 'bulge' in loop_type:
      return 0
   elif 'hairpin' in loop_type:
      if loop_length <= 3:
         return 0
      else:
         return 1 # one terminal mismatch term
   elif 'multi' in loop_type: #dangling end or terminal mismatch likely possible on each end of each helix
     #no_DE = len([i for i in loop_length.keys() if loop_length[i] > 0 or ((i == 0 and loop_length[max(loop_length.keys())] > 0) or (i > 0 and loop_length[i-1] > 0))])
     #no_DE = len([i for i in loop_length.keys() if loop_length[i] > 0])
     no_DE = len([i for i in loop_length.keys() if (i > 0 and (loop_length[i] > 0 or loop_length[i-1] > 0)) or (i == 0 and (loop_length[i] > 0 or loop_length[max(loop_length.keys())] > 0)) ])
     return no_DE
   elif 'exterior' in loop_type:
     # no_DE = len([i for i in loop_length.keys() if i > 0 and (loop_length[i] > 0 or loop_length[i-1] > 0)])
     #no_DE = len([i for i in loop_length.keys() if loop_length[i] > 0])
     no_DE = len([i for i in loop_length.keys() if i > 0 and (loop_length[i] > 0 or loop_length[i-1] > 0)])
     return no_DE      

def number_stacks_acroos_bulge(structure):
   """ one stack for each length-one bulge"""
   loop_types_and_len = get_roles_and_lengths_of_loops(structure)  
   bulges_coaxial_stacking = len([1 for loop_type, loop_length in loop_types_and_len if 'bulge' in loop_type and loop_length == 1])
   return bulges_coaxial_stacking 


###############################################################################################
# estimate neutral set sizes
###############################################################################################

def number_one_by_one_internal_loops(structure):
   loop_types_and_len = get_roles_and_lengths_of_loops(structure)
   number_one_by_one_internal = 0
   for loop_type, loop_length in loop_types_and_len:
      if loop_type.startswith('internal') and loop_length[0] == loop_length[1] == 1:
         number_one_by_one_internal += 1
   return number_one_by_one_internal


def energy_per_bp_to_bp_versatility(energy_per_base_pair):
   """interpolate between energy constraints and sequence constraints in stacks"""
   energy_list  = [-1 * G_max_bp, -0.5]
   v_list = [2/6.0, 1.0] 
   return np.interp(energy_per_base_pair, energy_list, v_list) 


def low_energy_set_size_estimate(structure, G_max=0):
   """use our new method for low-energy set size estimation with a cutoff of G_max"""
   assert G_max <= 10.0**(-5)
   nbp = structure.count('(')
   number_compatible_sequences = (4 ** (len(structure) - 2 * nbp)) * (6 ** nbp)
   N_stacking_end_of_loop = number_of_dangling_end_terminal_mismatch_terms(structure)
   G_const = loop_const_free_energy(structure) - 0.5 * N_stacking_end_of_loop - G_max  
   no_stacking_terms = nbp - cg.dotbracket_to_coarsegrained_lev2(structure).count('[') + number_stacks_acroos_bulge(structure) 
   energy_for_DE_TM = - G_const + G_max_bp * float(no_stacking_terms) + 0.2 * N_stacking_end_of_loop # how much energy cannot be covered by stacking
   if no_stacking_terms > 0:
      energy_per_base_pair = (-G_const)/float(no_stacking_terms) # how much energy must each base pair contribute
      f = energy_per_bp_to_bp_versatility(energy_per_base_pair) ** nbp #(nbp - number_isolated_bp) * (1/3.0) ** number_isolated_bp #isolated bps can only be GC
   else:
      f = (2/6.0)** nbp
   if energy_for_DE_TM < 0: #must be further constraints at unpaired sites: dangling ends (multiloops and exterior loops) and terminal mismatches (hairpin, internal, external, multiloop)
      if N_stacking_end_of_loop > 0:
         v_DETM = max(0.25, 1 - 0.75 * abs(energy_for_DE_TM)/(N_stacking_end_of_loop * 0.9))
         f = f * (v_DETM)**N_stacking_end_of_loop
      if abs(energy_for_DE_TM) - (N_stacking_end_of_loop * 0.9) > 0.05: # extra contraints on suboptimal structures (?)       
         f = 0.1 * f
   return number_compatible_sequences * f


def free_energy_estimate_with_heuristic(structure):
   return low_energy_set_size_estimate(structure, G_max= 1.5 - structure.count('(') - 2 *  cg.number_isolated_bp_nostackingterms(structure))


def low_enery_set_versatility_bp_estimate(structure, G_max=0):
   assert G_max <= 10.0**(-5)
   N_stacking_end_of_loop = number_of_dangling_end_terminal_mismatch_terms(structure)
   G_const = loop_const_free_energy(structure) - 0.5 * N_stacking_end_of_loop - G_max
   stacks = cg.dotbracket_to_coarsegrained_lev2(structure).count('[')
   no_stacking_terms = structure.count('(') - stacks + number_stacks_acroos_bulge(structure) 
   if no_stacking_terms > 0:
      energy_per_base_pair = (-G_const)/float(no_stacking_terms) # how much energy must each base pair contribute
      return energy_per_bp_to_bp_versatility(energy_per_base_pair)
   else:
      return 2/6.0

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   # test decomposition into loop types and lengths
   loop_types_and_len = get_roles_and_lengths_of_loops('..((.((....))((....))..))...')
   assert len([loop_type for loop_type, loop_length in loop_types_and_len if loop_type.startswith('multi')]) == 1
   assert len([loop_type for loop_type, loop_length in loop_types_and_len if loop_type.startswith('hairpin')]) == 2
   multiloop_elements = [loop_length for loop_type, loop_length in loop_types_and_len if loop_type.startswith('multi')][0]
   assert multiloop_elements[0] == 1 and multiloop_elements[1] == 0 and multiloop_elements[2] == 2
   exterior_elements = [loop_length for loop_type, loop_length in loop_types_and_len if loop_type.startswith('exterior')][0]
   assert exterior_elements[0] == 2 and exterior_elements[1] == 3
   #
   loop_types_and_len = get_roles_and_lengths_of_loops('...((....))((....)).....')
   assert len([loop_type for loop_type, loop_length in loop_types_and_len if loop_type.startswith('multi')]) == 0
   assert len([loop_type for loop_type, loop_length in loop_types_and_len if loop_type.startswith('hairpin')]) == 2
   exterior_elements = [loop_length for loop_type, loop_length in loop_types_and_len if loop_type.startswith('exterior')][0]
   assert exterior_elements[0] == 3 and exterior_elements[1] == 0 and exterior_elements[2] == 5
   #
   assert number_of_dangling_end_terminal_mismatch_in_specific_loop('multiloop', {1: 0, 2: 1, 0: 2}) == 2
   assert number_of_dangling_end_terminal_mismatch_in_specific_loop('exterior loop', {1: 0, 2: 1, 0: 2, 3: 5}) == 3

   # interpolation 
   assert abs(energy_per_bp_to_bp_versatility(-0.5) - 5.0/6) < 0.001
   assert abs(energy_per_bp_to_bp_versatility(-5) - 2.0/6) < 0.001

