
from AlgorithmType import AlgorithmType

from Combo_Seg import Combo_Seg
from EFirst import EFirst
from EGreedy import EGreedy
from EGreedy_Annealing import EGreedy_Annealing
from EGreedy_Disjoint import EGreedy_Disjoint
from EGreedy_Hybrid import EGreedy_Hybrid
from EGreedy_Lin import EGreedy_Lin
from EGreedy_Lin_Hybrid import EGreedy_Lin_Hybrid
from EGreedy_TS import EGreedy_TS
from LinUCB_Disjoint import LinUCB_Disjoint
from LinUCB_GP import LinUCB_GP
from LinUCB_GP_All import LinUCB_GP_All
from LinUCB_Hybrid import LinUCB_Hybrid 
from NN import NN
from TS import TS
from TS_Bootstrap import TS_Bootstrap
from TS_Lin import TS_Lin
from TS_Disjoint import TS_Disjoint
from TS_Hybrid import TS_Hybrid
from TS_Laplace import TS_Laplace
from TS_Truncated import TS_Truncated
from TS_Gibbs import TS_Gibbs
from TS_RLR import TS_RLR
from UCB import UCB
from Random import Random
from Ensemble import Ensemble

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

		elif algorithm_type == AlgorithmType.EGreedy_Lin:
			return EGreedy_Lin(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Seg_Lin:
			return Combo_Seg(alpha, AlgorithmType.EGreedy_Lin)

		elif algorithm_type == AlgorithmType.EGreedy_Lin_Hybrid:
			return EGreedy_Lin_Hybrid(alpha)

		elif algorithm_type == AlgorithmType.TS:
			return TS(alpha)

		elif algorithm_type == AlgorithmType.TS_Bootstrap:
			return TS_Bootstrap(alpha)
				
		elif algorithm_type == AlgorithmType.TS_Lin:
			return TS_Lin(alpha)

		elif algorithm_type == AlgorithmType.TS_Seg:
			return Combo_Seg(alpha, AlgorithmType.TS)

		elif algorithm_type == AlgorithmType.TS_Disjoint:
			return TS_Disjoint(alpha)

		elif algorithm_type == AlgorithmType.TS_Hybrid:
			return TS_Hybrid(alpha)

		elif algorithm_type == AlgorithmType.TS_Truncated:
			return TS_Truncated(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_TS:
			return EGreedy_TS(alpha)

		elif algorithm_type == AlgorithmType.TS_Gibbs:
			return TS_Gibbs(alpha)
			
		elif algorithm_type == AlgorithmType.TS_Laplace:
			return TS_Laplace(alpha)

		elif algorithm_type == AlgorithmType.EGreedy_Annealing:
			return EGreedy_Annealing(alpha)

		elif algorithm_type == AlgorithmType.NN:
			return NN(alpha)

		elif algorithm_type == AlgorithmType.Ensemble:
			return Ensemble(alpha)

		elif algorithm_type == AlgorithmType.TS_RLR:
			return TS_RLR(alpha)
		
		else:
			raise NotImplementedError("Non-implemented algorithm." + algorithm_type.name)
