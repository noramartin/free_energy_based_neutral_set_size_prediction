import networkx as nx
import string

def number_isolated_bp(structure):
   return len([c for indexc, c in enumerate(structure) if c == '(' and find_element_length(indexc, structure) == 1]) 

def number_isolated_bp_nostackingterms(structure):
   n_isolated = 0
   bp_pos = get_basepair_indices_from_dotbracket(structure)
   for indexc, c in enumerate(structure):
      if c == '(' and find_element_length(indexc, structure) == 1:
         adjacant_bps = get_previous_and_next_basepair_pos(indexc, structure)
         if (adjacant_bps[0] != ' ' and abs(adjacant_bps[0] - indexc) + abs(bp_pos[adjacant_bps[0]] - bp_pos[indexc]) <= 3) or (adjacant_bps[1] != ' ' and abs(adjacant_bps[1] - indexc) + abs(bp_pos[adjacant_bps[1]] - bp_pos[indexc]) <= 3):
            pass
         else:
            n_isolated += 1
   return n_isolated

############################################################################################################
## manipulate structures
############################################################################################################

def cut_ends(dotbracketstring):
   """cut unpaired regions at either end of dotbacketstring"""
   dotbracketstring = dotbracketstring.strip('._')
   if dotbracketstring:
      return dotbracketstring
   else:
      return '.' #for unfolded structure

def dotbracket_to_coarsegrained_lev2(dotbracketstring):
   """transform a full dotbracket representation to a coarse-grained 
   (type 1 as defined in Janssen, Reeder, Giegerich (2008). BMC bioinformatics)"""
   fine_grained_to_coarse_grained_symbol = {'(': '[', ')': ']', '.': '_'}
   dotbracketstring = cut_ends(dotbracketstring)
   tree_secondarystr, basepair_index_mapping = get_tree_bpmap_from_dotbracket(dotbracketstring)
   coarse_grained_string_list = []
   for charindex, char in enumerate(dotbracketstring):
      if charindex == 0  or dotbracketstring[charindex-1] != dotbracketstring[charindex]:
         if dotbracketstring[charindex] != '.': 
            coarse_grained_string_list.append(fine_grained_to_coarse_grained_symbol[char])
         elif not is_multiloop_from_tree(charindex, tree_secondarystr, dotbracketstring) and not is_exterior_loop(charindex, dotbracketstring):
            coarse_grained_string_list.append(fine_grained_to_coarse_grained_symbol[char])
      elif dotbracketstring[charindex-1] == dotbracketstring[charindex] and dotbracketstring[charindex] != '.': #two subsequent brackets of same type
         if not abs(basepair_index_mapping[charindex]-basepair_index_mapping[charindex-1])<1.5:
            coarse_grained_string_list.append(fine_grained_to_coarse_grained_symbol[char])
   if len(coarse_grained_string_list) > 0:
      return remove_hairpins_from_cg(coarse_grained_string_list)
   else:
      return '_'


def remove_hairpins_from_cg(cg_list):
   #cg_bp_map = get_basepair_indices_from_dotbracket(coarse_grained_string)
   #hairpin_indices = set([(i1+i2)//2 for i1, i2 in cg_bp_map.iteritems() if abs(i1-i2) == 2])
   hairpin_indices = [i-1 for i in range(2, len(cg_list)) if cg_list[i] == ']' and cg_list[i-1] == '_' and cg_list[i-2] == '[']
   return ''.join([c for i, c in enumerate(cg_list) if i not in hairpin_indices])

############################################################################################################
## find lengths of structural elements 
############################################################################################################

def find_len_of_stack(pos, dotbracketstring):
   """ return the length of the stack at position pos in the structure given by the dot-bracket string
   bulges are defined as the end of the stack on both strands, i.e. if a base pair is at i, j, the base pair at i+1, j-2 would not belong to the same stack"""
   basepair_index_mapping = get_basepair_indices_from_dotbracket(dotbracketstring)
   assert pos in basepair_index_mapping
   node_of_basepair = min(pos, basepair_index_mapping[pos])
   ## make network of basepairs and connect adjacent basepairs in stacks - then stack size is size of component
   base_pair_neighbour_graph = nx.Graph()
   base_pair_neighbour_graph.add_nodes_from(set([min(b) for b in basepair_index_mapping.iteritems()])) # each base pair is represented by the pos of the opening bracket
   for b1, b2 in basepair_index_mapping.iteritems():
      for a1, a2 in basepair_index_mapping.iteritems():
         if b1<b2 and a1<a2: # both ordered from ( to )
            if b1 != a1: # distinct bas pairs
               if abs(b1-a1) == 1 and abs(b2-a2) == 1:
                  base_pair_neighbour_graph.add_edge(a1, b1)
   return len(nx.node_connected_component(base_pair_neighbour_graph, node_of_basepair))


def find_len_of_loop(pos, dotbracketstring):
   """ find length of the unpaired region that pos is part of"""
   assert dotbracketstring[pos] =='.'
   previous_bp_index = max([index for index, c in enumerate(dotbracketstring) if c in ['(', ')'] and index < pos]+[-1])
   next_bp_index = min([index for index, c in enumerate(dotbracketstring) if c in ['(', ')']and index > pos]+[len(dotbracketstring)])
   return next_bp_index - previous_bp_index - 1

def find_element_length(pos, dotbracketstring):
   """ find length of the unpaired region or stack that pos is part of"""
   if dotbracketstring[pos] == '.':
      return find_len_of_loop(pos, dotbracketstring)
   else:
      return find_len_of_stack(pos, dotbracketstring)
############################################################################################################
## extract base pairs
############################################################################################################

def get_basepair_indices_from_dotbracket(dotbracketstring):
   """extract a dictionary mapping each paired position with its partner:
   each base pair is represented twice: mapping from opening to closing bracket and vice versa"""
   base_pair_mapping = {}
   number_open_brackets = 0
   opening_level_vs_index = {}
   for charindex, char in enumerate(dotbracketstring):
      if char in ['(', '[']:
         number_open_brackets += 1
         opening_level_vs_index[number_open_brackets] = charindex
      elif char in [')', ']']:
         base_pair_mapping[charindex] = opening_level_vs_index[number_open_brackets]
         base_pair_mapping[opening_level_vs_index[number_open_brackets]] = charindex
         del opening_level_vs_index[number_open_brackets]
         number_open_brackets -= 1
      elif char in ['.', '_']:
         pass
      else:
         raise ValueError('invalid character in dot-bracket string')
      if number_open_brackets < 0:
         raise ValueError('invalid dot-bracket string')
   if number_open_brackets != 0:
      raise ValueError('invalid dot-bracket string')
   return base_pair_mapping

############################################################################################################
## classify structural contexts
############################################################################################################

def is_multiloop_from_tree(pos, tree_secondarystr, dotbracketstring):
   """is the unpaired site at pos a multiloop;
   which means an loop with one 'parent' stem and more than one 'child' stem
   this is tested by converting the secondary structure graph to a tree"""
   assert dotbracketstring[pos] in ['.', '_']
   parent_stem = [p for p in tree_secondarystr.predecessors(pos)][0]
   number_stems_below_parent_stem = len([x for x in tree_secondarystr.successors(parent_stem) if len([y for y in tree_secondarystr.successors(x)])>0])
   if number_stems_below_parent_stem > 1:
      return True
   else:
      return False

def get_char_before_and_after_pos(pos, dotbracketstring):
   """return the character before and after the given position pos in the dot-bracket string;
   if pos is at either end of the string, return ' ' """
   if pos == 0:
      return [' ', dotbracketstring[pos+1]]
   elif pos == len(dotbracketstring) - 1:
      return [dotbracketstring[pos-1], ' ']   
   else:
      return [dotbracketstring[pos-1], dotbracketstring[pos+1]]   

def get_previous_and_next_basepair_char(pos, dotbracketstring):
   """return the next and previous base pair character in the string, 
   starting from the given position pos;
   if no base pair is found before/after, return ' ' instead"""
   if dotbracketstring[pos+1:].count(')') + dotbracketstring[pos+1:].count(']')  == 0:
      next_char = ' '
   else:
      next_char = [c for c in dotbracketstring[pos+1:] if c not in ['.' ,'_']][0]
   if dotbracketstring[:pos].count('(') + dotbracketstring[:pos].count('[') == 0:
      prev_char = ' '
   else:
      prev_char = [c for c in dotbracketstring[:pos] if c not in ['.' ,'_']][-1]   
   return [prev_char, next_char]

def get_previous_and_next_basepair_pos(pos, dotbracketstring):
   """return the next and previous base pair site in the string, 
   starting from the given position pos;
   if no base pair is found before/after, return np.nan instead"""
   if dotbracketstring[pos+1:].count(')') + dotbracketstring[pos+1:].count(']')  == 0:
      next_pos = ' '
   else:
      next_pos = [i_c for i_c, c in enumerate(dotbracketstring) if i_c > pos and c not in ['.' ,'_']][0]
   if dotbracketstring[:pos].count('(') + dotbracketstring[:pos].count('[') == 0:
      prev_pos = ' '
   else:
      prev_pos = [i_c for i_c, c in enumerate(dotbracketstring) if i_c < pos and c not in ['.' ,'_']][-1]   
   return [prev_pos, next_pos]



def find_role_in_structure(pos, dotbracketstring):
   """classify structural role of site at pos;
   returns type of structural element first and second whether middle/end"""
   char_before_and_after_pos = get_char_before_and_after_pos(pos, dotbracketstring)
   previous_and_next_basepair_char = get_previous_and_next_basepair_char(pos, dotbracketstring)
   if dotbracketstring[pos] == '(' or dotbracketstring[pos] == ')':
      basepairing = get_basepair_indices_from_dotbracket(dotbracketstring)
      paired_site = basepairing[pos]
      char_before_and_after_bp = char_before_and_after_pos + get_char_before_and_after_pos(paired_site, dotbracketstring)
      if char_before_and_after_bp.count('.') + char_before_and_after_bp.count(' ') > 0:
         return 'stack', 'end'
      elif abs(basepairing[pos-1] - paired_site) == 1 and abs(basepairing[pos+1] - paired_site) == 1:
         return 'stack', 'middle'
      else:
         return 'stack', 'end' # neighbours not paired with neighbours of paired site (can happen e.g. at three-way junction)
   elif is_exterior_loop(pos, dotbracketstring):
      if '(' in char_before_and_after_pos or ')' in char_before_and_after_pos or pos == 0 or pos==len(dotbracketstring)-1:
         return 'exterior loop', 'end'
      else:
         return 'exterior loop', 'middle'
   elif previous_and_next_basepair_char[0] == '(' and previous_and_next_basepair_char[1] == ')':
      if '(' not in char_before_and_after_pos and ')' not in char_before_and_after_pos:
         return 'hairpin loop', 'middle'
      else:
         return 'hairpin loop', 'end'
   elif is_multiloop(pos, dotbracketstring):
      if '(' in char_before_and_after_pos or ')' in char_before_and_after_pos:
         return 'multiloop', 'end'
      else:
         return 'multiloop', 'middle'
   elif previous_and_next_basepair_char[0] == previous_and_next_basepair_char[1]:
      prev_bp_pos, next_bp_pos = get_previous_and_next_basepair_pos(pos, dotbracketstring)
      basepairing = get_basepair_indices_from_dotbracket(dotbracketstring)
      isinternalloop = abs(basepairing[next_bp_pos] - basepairing[prev_bp_pos]) - 1
      if isinternalloop and '(' not in char_before_and_after_pos and ')' not in char_before_and_after_pos:
         return 'internal loop', 'middle'
      elif isinternalloop:
         return 'internal loop', 'end'
      elif '(' not in char_before_and_after_pos and ')' not in char_before_and_after_pos:
         return 'bulge', 'middle'
      else:
         return 'bulge', 'end'
   else:
      raise ValueError('Unknown structural element.')

def return_length_of_both_sides_internal_loop(pos, dotbracketstring):
   assert find_role_in_structure(pos, dotbracketstring)[0].startswith('internal')
   char_before_and_after_pos = get_char_before_and_after_pos(pos, dotbracketstring)
   previous_and_next_basepair_char = get_previous_and_next_basepair_char(pos, dotbracketstring)
   prev_bp_pos, next_bp_pos = get_previous_and_next_basepair_pos(pos, dotbracketstring)
   basepairing = get_basepair_indices_from_dotbracket(dotbracketstring)
   loop_length_this_side = abs(next_bp_pos - prev_bp_pos) - 1
   loop_length_other_side = abs(basepairing[next_bp_pos] - basepairing[prev_bp_pos]) - 1
   return loop_length_this_side, loop_length_other_side


def is_exterior_loop(pos, dotbracketstring):
   """is the unpaired site at pos an exterior loop;
   which means either an unpaired stretch at either end of the chain
   or an unpaired region, which is not enclosed within the span of any base pair"""
   assert dotbracketstring[pos] == '.'
   if '(' not in dotbracketstring[:pos] or ')' not in dotbracketstring[pos:]:
      return True 
   elif dotbracketstring[:pos].count('(') == dotbracketstring[:pos].count(')'):
      return True 
   else:
      return False

def return_complete_multiloop(pos, dotbracketstring):
   assert is_multiloop(pos, dotbracketstring)
   tree_secondarystr = get_tree_bpmap_from_dotbracket(dotbracketstring)[0]
   parent_stem = [p for p in tree_secondarystr.predecessors(pos)][0]
   complete_multiloop = [x for x in tree_secondarystr.successors(parent_stem) if len([y for y in tree_secondarystr.successors(x)]) == 0]
   stems_below_parent_stem = sorted([x for x in tree_secondarystr.successors(parent_stem) if len([y for y in tree_secondarystr.successors(x)])>0])
   branch_stem_vs_multiloop_sites_before = {i: [s for s in complete_multiloop if s < x and (i == 0 or s > stems_below_parent_stem[i-1])] for i, x in enumerate(stems_below_parent_stem)}
   branch_stem_vs_multiloop_sites_before[len(stems_below_parent_stem)] = [s for s in complete_multiloop if s > max(stems_below_parent_stem)]
   return branch_stem_vs_multiloop_sites_before, complete_multiloop

def is_multiloop(pos, dotbracketstring):
   """is the unpaired site at pos a multiloop;
   which means an loop with one 'parent' stem and more than one 'child' stem
   this is tested by converting the secondary structure graph to a tree"""
   assert dotbracketstring[pos] == '.'
   tree_secondarystr = get_tree_bpmap_from_dotbracket(dotbracketstring)[0]
   parent_stem = [p for p in tree_secondarystr.predecessors(pos)][0]
   number_stems_below_parent_stem = len([x for x in tree_secondarystr.successors(parent_stem) if len([y for y in tree_secondarystr.successors(x)])>0])
   if number_stems_below_parent_stem > 1:
      return True
   else:
      return False


def get_tree_bpmap_from_dotbracket(dotbracketstring):
   """converting the secondary structure graph to a tree;
   each site is represented by a node whose integer reference is the site's sequence position;
   the tree is grounded in an additional node, '-1';
   each node is a child node of the next enclosing base pair"""
   tree, bp_mapping, current_node = nx.DiGraph(), {}, -1
   tree.add_node(current_node)
   for charindex, char in enumerate(dotbracketstring):
      if char in ['(', '[']:
         tree.add_edge(current_node, charindex)
         current_node = charindex
         if char == '[' and dotbracketstring[charindex+1] == ']':
            tree.add_edge(current_node, charindex + len(dotbracketstring) + 1) # add hairpin if not included
      elif char in [')', ']']:
         bp_mapping[charindex] = current_node
         bp_mapping[current_node] = charindex
         current_node = [n for n in tree.predecessors(current_node)][0]
      elif char in ['.', '_']:
         tree.add_edge(current_node, charindex)
      else:
         raise ValueError('invalid character in dot-bracket string')
   return tree, bp_mapping


############################################################################################################
## check viability of structure
############################################################################################################

def sequence_compatible_with_basepairs(sequence, structure):
   """check if the input sequence (string containing AUGC) is 
   compatibale with the dot-bracket input structure,
   i.e. paired sites are a Watson-Crick pair or GU"""
   allowed_basepairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
   for b in sequence:
      assert b in ['A', 'U', 'G', 'C']
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.iteritems():
      if sequence[baseindex1]+sequence[baseindex2] not in allowed_basepairs:
         return False
   return True

def hairpin_loops_long_enough(structure):
   """check if any paired sites in the dot-bracket input structure
   are at least four sites apart"""
   bp_mapping = get_basepair_indices_from_dotbracket(structure)
   for baseindex1, baseindex2 in bp_mapping.iteritems():
      if abs(baseindex2-baseindex1) < 4:
         return False
   return True



def is_likely_to_be_valid_structure(structure, allow_isolated_bps=False):
   """tests if a structure in dotbracket format is likely to be a valid structure:
   basepairs closed, length of hairpin loops (>3), presence of basepairs and optionally isolated base pairs"""
   if not basepairs_closed(structure):
      return False
   if not hairpin_loops_long_enough(structure):
      return False
   if not structure.count(')') > 0:
      return False
   if not allow_isolated_bps and has_length_one_stack(structure):
      return False
   else:
      return True


def basepairs_closed(structure):
   """test if all brackets are closed correctly in a dot-bracket string"""
   try:
      bp_map = get_basepair_indices_from_dotbracket(structure)
      return True
   except (ValueError, KeyError):
      return False

def has_length_one_stack(dotbracketstring):
   """test if dotbracketstring has isolated base pairs"""
   for pos, char in enumerate(dotbracketstring):
      if char in [')', '('] and find_len_of_stack(pos, dotbracketstring) < 2:
         return 1
   return 0

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   teststructure1 = '...(((..((...)))))..((....)).'
   teststructure2 = '(((...))).'
   ### test cutting ends
   assert cut_ends(teststructure1) == '(((..((...)))))..((....))'
   assert cut_ends(teststructure2) == '(((...)))'
   ### test stack of length one
   assert has_length_one_stack('.((.(...)))..') and has_length_one_stack('.((.(...).))..') and has_length_one_stack('.((.(...).))..')
   assert has_length_one_stack('.(...)...')
   assert not (has_length_one_stack(teststructure1) or has_length_one_stack(teststructure2))
   ### convert to coarsegrained structure
   assert dotbracket_to_coarsegrained_lev2(teststructure1) == '[_[]][]'
   assert dotbracket_to_coarsegrained_lev2(teststructure2) == '[]'
   assert dotbracket_to_coarsegrained_lev2('.((.(...)))..') == '[_[]]' and dotbracket_to_coarsegrained_lev2('.((.(...).))..') == '[_[]_]' and dotbracket_to_coarsegrained_lev2('.(((...).))..') == '[[]_]'
   ### test stack lengths
   assert find_len_of_stack(3, teststructure1) == find_len_of_stack(4, teststructure1) == find_len_of_stack(5, teststructure1) == find_len_of_stack(15, teststructure1) == find_len_of_stack(16, teststructure1) == 3
   assert find_len_of_stack(8, teststructure1) == find_len_of_stack(9, teststructure1) == find_len_of_stack(13, teststructure1) == find_len_of_stack(14, teststructure1) == 2
   assert find_len_of_stack(0, teststructure2) == find_len_of_stack(6, teststructure2) == find_len_of_stack(7, teststructure2) == find_len_of_stack(8, teststructure2) == 3
   ### test loop lengths
   assert find_len_of_loop(0, teststructure1) == find_len_of_loop(1, teststructure1) == find_len_of_loop(2, teststructure1) == 3
   assert find_len_of_loop(6, teststructure1) == find_len_of_loop(7, teststructure1) == 2
   assert find_len_of_loop(9, teststructure2) == 1
   ### test base pair extraction
   bp_mapping1 = get_basepair_indices_from_dotbracket(teststructure1)
   assert bp_mapping1[3] == 17 and bp_mapping1[4] == 16 and bp_mapping1[5] == 15 and bp_mapping1[8] == 14 and bp_mapping1[20] == 27
   assert bp_mapping1[17] == 3 and bp_mapping1[16] == 4 and bp_mapping1[15] == 5 and bp_mapping1[14] == 8 and bp_mapping1[27] == 20
   bp_mapping2 = get_basepair_indices_from_dotbracket(teststructure2)
   assert bp_mapping2[0] == 8 and bp_mapping2[1] == 7 and bp_mapping2[2] == 6 
   ### test get_char_before_and_after_pos
   assert get_char_before_and_after_pos(3, teststructure1)[0] == '.' and get_char_before_and_after_pos(3, teststructure1)[1] == '('
   assert get_char_before_and_after_pos(0, teststructure1)[0] == ' ' and get_char_before_and_after_pos(0, teststructure1)[1] == '.'
   assert get_char_before_and_after_pos(9, teststructure2)[0] == ')' and get_char_before_and_after_pos(9, teststructure2)[1] == ' '
   ### test get_previous_and_next_basepair_char
   assert get_previous_and_next_basepair_char(2, teststructure1)[0] == ' ' and get_previous_and_next_basepair_char(2, teststructure1)[1] == '('
   assert get_previous_and_next_basepair_char(6, teststructure1)[0] == '(' and get_previous_and_next_basepair_char(6, teststructure1)[1] == '('
   assert get_previous_and_next_basepair_char(10, teststructure1)[0] == '(' and get_previous_and_next_basepair_char(10, teststructure1)[1] == ')'
   assert get_previous_and_next_basepair_char(19, teststructure1)[0] == ')' and get_previous_and_next_basepair_char(19, teststructure1)[1] == '('
   assert get_previous_and_next_basepair_char(22, teststructure1)[0] == '(' and get_previous_and_next_basepair_char(22, teststructure1)[1] == ')'
   assert get_previous_and_next_basepair_char(9, teststructure2)[0] == ')' and get_previous_and_next_basepair_char(9, teststructure2)[1] == ' '
   ### test find_role_in_structure
   teststructure1 = '...(((..((...)))))..'
   structural_labels = [('exterior loop', 'end'), ('exterior loop', 'middle'), ('exterior loop', 'end'),
                        ('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('bulge', 'end'), ('bulge', 'end'),
                        ('stack', 'end'), ('stack', 'end'), ('hairpin loop', 'end'),
                        ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure1)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure1)[1] == structure_description[1]
   ### test find_role_in_structure for multiloop case
   teststructure2 = '(((..((...))..((...)))))..'
   structural_labels = [('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('multiloop', 'end'), ('multiloop', 'end'),
                        ('stack', 'end'), ('stack', 'end'), ('hairpin loop', 'end'),
                        ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('multiloop', 'end'), ('multiloop', 'end'), 
                        ('stack', 'end'), ('stack', 'end'), ('hairpin loop', 'end'),
                        ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure2)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure2)[1] == structure_description[1]
   multiloop_example2 = '((((..(((....))).(((....)))..))))..'
   assert 'multiloop' in [find_role_in_structure(charindex, multiloop_example2)[0] for charindex in range(len(multiloop_example2))]
   ### test find_role_in_structure for internal loop case
   teststructure3 = '(((..((...))...)))..'
   structural_labels = [('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('internal loop', 'end'), ('internal loop', 'end'),
                        ('stack', 'end'), ('stack', 'end'), ('hairpin loop', 'end'),
                        ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('internal loop', 'end'), ('internal loop', 'middle'), ('internal loop', 'end'),
                        ('stack', 'end'), ('stack', 'middle'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure3)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure3)[1] == structure_description[1]
   teststructure4 = '.((.(...)))..'
   structural_labels = [('exterior loop', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('bulge', 'end'), ('stack', 'end'),
                        ('hairpin loop', 'end'), ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure4)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure4)[1] == structure_description[1]
   teststructure5 = '.((.(...).))..'
   structural_labels = [('exterior loop', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('internal loop', 'end'), ('stack', 'end'),
                        ('hairpin loop', 'end'), ('hairpin loop', 'middle'), ('hairpin loop', 'end'), ('stack', 'end'),
                        ('internal loop', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure5)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure5)[1] == structure_description[1]
   teststructure6 = '.((((...))((...))))..'
   structural_labels = [('exterior loop', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('stack', 'end'),
                        ('hairpin loop', 'end'), ('hairpin loop', 'middle'), ('hairpin loop', 'end'), 
                        ('stack', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('stack', 'end'), ('hairpin loop', 'end'), ('hairpin loop', 'middle'), ('hairpin loop', 'end'), 
                        ('stack', 'end'),('stack', 'end'), ('stack', 'end'), ('stack', 'end'),
                        ('exterior loop', 'end'), ('exterior loop', 'end')]
   for charindex, structure_description in enumerate(structural_labels):
      assert find_role_in_structure(charindex, teststructure6)[0] == structure_description[0] and find_role_in_structure(charindex, teststructure6)[1] == structure_description[1]

   ### test sequence-structure compatibility
   teststructure1 = '...(((..((...)))))..((....)).'
   teststructure2 = '(((...))).'
   assert sequence_compatible_with_basepairs('AAGGGCCAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert not sequence_compatible_with_basepairs('AAGUGCCAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert sequence_compatible_with_basepairs('AAGGGCAAGGAUUCCGCCGAGCGAUAGCA', teststructure1)
   assert sequence_compatible_with_basepairs('GUACGUUGC', teststructure2)
   assert not sequence_compatible_with_basepairs('GUACGUGGC', teststructure2)
   assert sequence_compatible_with_basepairs('GUACAUUGU', teststructure2)
   ### test length of hairpin loops
   assert hairpin_loops_long_enough(teststructure1) and hairpin_loops_long_enough(teststructure2)
   assert not hairpin_loops_long_enough('...(((..((..)))))..((....)).')
   ## test whether invalid structures are recognised correctly
   assert is_likely_to_be_valid_structure('...((..((...))))..((...))..', allow_isolated_bps=False)
   assert is_likely_to_be_valid_structure('...((..((...))))..(...)..', allow_isolated_bps=True) and not is_likely_to_be_valid_structure('...((..((...))))..(...)..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...((..((....)))))..((...)..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...((..((..))))..((...))..', allow_isolated_bps=False)
   assert not is_likely_to_be_valid_structure('...(((..((...))))..((...))..', allow_isolated_bps=False)
   ## test whether multiloop returned correctly
   teststructure = '..(((..((.....))...((...))..(((.....))).)))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(5, teststructure)
   assert len(complete_multiloop) == 8
   for MLsite in [5, 6, 16, 17, 18, 26, 27, 39]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 2 and len(branch_stem_vs_multiloop_sites_before[1]) == 3 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 1
   teststructure = '....(((((.....))...((...))..(((.....))).)))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(26, teststructure)
   assert len(complete_multiloop) == 6
   for MLsite in [16, 17, 18, 26, 27, 39]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 0 and len(branch_stem_vs_multiloop_sites_before[1]) == 3 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 1
   teststructure = '....(((((.....))...((...))..(((.....))))))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(26, teststructure)
   assert len(complete_multiloop) == 5
   for MLsite in [16, 17, 18, 26, 27]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 0 and len(branch_stem_vs_multiloop_sites_before[1]) == 3 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 0
   teststructure = '..(((..((.....))...((...))..(((.....))))))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(18, teststructure)
   assert len(complete_multiloop) == 7
   for MLsite in [5, 6, 16, 17, 18, 26, 27]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 2 and len(branch_stem_vs_multiloop_sites_before[1]) == 3 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 0
   teststructure = '..(((..((.....))((......))..(((.....))).)))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(27, teststructure)
   assert len(complete_multiloop) == 5
   for MLsite in [5, 6, 26, 27, 39]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 2 and len(branch_stem_vs_multiloop_sites_before[1]) == 0 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 1
   teststructure = '..(((..(.......)((......))..(((.....))).)))..'
   branch_stem_vs_multiloop_sites_before, complete_multiloop = return_complete_multiloop(27, teststructure)
   assert len(complete_multiloop) == 5
   for MLsite in [5, 6, 26, 27, 39]:
      assert MLsite in complete_multiloop
   assert len(branch_stem_vs_multiloop_sites_before[0]) == 2 and len(branch_stem_vs_multiloop_sites_before[1]) == 0 and len(branch_stem_vs_multiloop_sites_before[2]) == 2 and len(branch_stem_vs_multiloop_sites_before[3]) == 1

   # test whether internal loops measured correctly
   assert return_length_of_both_sides_internal_loop(3, '.((.(((...)))..))') == (1, 2)
   assert return_length_of_both_sides_internal_loop(8, '..(((..(.(((...))).)...((...))..(((.....))))))..') == (1, 1)
   assert return_length_of_both_sides_internal_loop(5, '.((......(((...)))...))') == (6, 3)
   print 'test completed'
   ## test number of isolate bps
   assert number_isolated_bp('((..((....)).))') == 0
   assert number_isolated_bp('((..(....).))') == 1
   assert number_isolated_bp('((..(....)))') == 1
   assert number_isolated_bp('(((....)..))') == 1
   assert number_isolated_bp('((....)..)') == 2
   

