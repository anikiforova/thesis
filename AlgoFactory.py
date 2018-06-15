
from AlgorithmType import AlgorithmType

from Combo_Seg import Combo_Seg
from EFirst import EFirst
from EGreedy import EGreedy
from EGreedy_Disjoint import EGreedy_Disjoint
from EGreedy_Hybrid import EGreedy_Hybrid
from EGreedy_Lin import EGreedy_Lin
from EGreedy_Lin_Hybrid import EGreedy_Lin_Hybrid
from LinUCB_Disjoint import LinUCB_Disjoint
from LinUCB_GP import LinUCB_GP
from LinUCB_GP_All import LinUCB_GP_All
from LinUCB_SGP import LinUCB_SGP
from LinUCB_Hybrid import LinUCB_Hybrid 
from UCB import UCB
from Random import Random

from Util import to_vector

class AlgoFactory:

	def get_algorithm(algorithm_type, alpha):
		if algorithm_type == AlgorithmType.Random:
			return Random(alpha)

		elif algorithm_type == AlgorithmType.EFirst:
			return EFirst(alpha)

		elif algorithm_type == AlgorithmType.EGreedy:
			return EGreedy(alpha) 

		elif algorithm_type == AlgorithmType.EGreedy_Disjoint:
			return EGreedy_Disjoint(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Hybrid:
			return EGreedy_Hybrid(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Seg:
			return Combo_Seg(alpha, AlgorithmType.EGreedy)

		elif algorithm_type == AlgorithmType.LinUCB_Disjoint:
			return LinUCB_Disjoint(alpha)

		elif algorithm_type == AlgorithmType.LinUCB_GP:
			return LinUCB_GP(alpha)

		elif algorithm_type == AlgorithmType.LinUCB_GP_All:
			return LinUCB_GP_All(alpha)
		
		elif algorithm_type == AlgorithmType.LinUCB_Hybrid:
			return LinUCB_Hybrid(alpha)

		elif algorithm_type == AlgorithmType.UCB:
			return UCB(alpha)

		elif algorithm_type == AlgorithmType.UCB_Seg:
			return Combo_Seg(alpha, AlgorithmType.UCB)

		elif algorithm_type == AlgorithmType.LinUCB_SGP:
			return LinUCB_SGP(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Lin:
			return EGreedy_Lin(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Seg_Lin:
			return Combo_Seg(alpha, AlgorithmType.EGreedy_Lin)

		elif algorithm_type == AlgorithmType.EGreedy_Lin_Hybrid:
			return EGreedy_Lin_Hybrid(alpha)
			
		else:
			raise NotImplementedError("Non-implemented algorithm." + algorithm_type.name)
