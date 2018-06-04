
from AlgorithmType import AlgorithmType

# from Combo_Seg import Combo_Seg
# from EFirst import EFirst
from EGreedy import EGreedy
from EGreedy_Disjoint import EGreedy_Disjoint
# from EGreedy_Hybrid import EGreedy_Hybrid
# from LinUCB_Disjoint import LinUCB_Disjoint
# from LinUCB_GP import LinUCB_GP
from LinUCB_GP_All import LinUCB_GP_All
# from LinUCB_Hybrid import LinUCB_Hybrid 
from UCB import UCB
from Random import Random

class AlgoFactory:

	def get_algorithm(algorithm_type, alpha, clusters):
		if algorithm_type == AlgorithmType.Random:
			return Random(alpha, clusters)

		# elif algorithm_type == AlgorithmType.EFirst:
		# 	return EFirst(alpha)

		elif algorithm_type == AlgorithmType.EGreedy:
			return EGreedy(alpha, clusters) 

		elif algorithm_type == AlgorithmType.EGreedy_Disjoint:
			return EGreedy_Disjoint(alpha)

		# elif algorithm_type == AlgorithmType.EGreedy_Hybrid:
		# 	return EGreedy_Hybrid(alpha)

		# elif algorithm_type == AlgorithmType.EGreedy_Seg:
		# 	return Combo_Seg(alpha, AlgorithmType.EGreedy)

		# elif algorithm_type == AlgorithmType.LinUCB_Disjoint:
		# 	return LinUCB_Disjoint(alpha)

		# elif algorithm_type == AlgorithmType.LinUCB_GP:
		# 	return LinUCB_GP(alpha)

		elif algorithm_type == AlgorithmType.LinUCB_GP_All:
			return LinUCB_GP_All(alpha, clusters)
		
		# elif algorithm_type == AlgorithmType.LinUCB_Hybrid:
			# return LinUCB_Hybrid(alpha)

		elif algorithm_type == AlgorithmType.UCB:
			return UCB(alpha, clusters)

		# elif algorithm_type == AlgorithmType.UCB_Seg:
		# 	return Combo_Seg(alpha, AlgorithmType.UCB)
			
		else:
			raise NotImplementedError("Non-implemented algorithm." + algorithm_type.name)
