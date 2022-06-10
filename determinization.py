import numpy as np
import itertools

from pddlgym.parser import Operator
from pddlgym.structs import ground_literal, Literal, LiteralConjunction, ProbabilisticEffect

def compute_determinizations(learned_operators):

    for operator in learned_operators:
        lifted_effects = operator.effects
        probs = []
        all_effects = []
        all_literal_possibilities = []
        for lifted_effect in lifted_effects.literals:
            # If this effect is probabilistic...
            if isinstance(lifted_effect, ProbabilisticEffect):
                literals = lifted_effect.literals
                probs_literals = lifted_effect.probabilities
                all_literal_possibilities.append(list(zip(probs_literals, literals)))
            else:
                all_literal_possibilities.append([(1.0, lifted_effect)])
        all_paths_literal_possibilities = list(itertools.product(*all_literal_possibilities))
        for path in all_paths_literal_possibilities:
            probs_path, literals_path = zip(*path)
            probs.append(np.prod(probs_path))
            all_effects.append(literals_path)
    
