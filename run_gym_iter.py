from pddlgym.structs import ground_literal, Literal, LiteralConjunction, ProbabilisticEffect
from pddlgym.inference import find_satisfying_assignments
import numpy as np
import itertools

def get_next_state(state, action, operators):
    """Given a state and action, returns the MOST LIKELY next state
    under the given set of operators (the given learned model).
    """
    # Step 1: Go through each operator. Try to match its preconditions with
    # the given state. If that doesn't work, continue to the next operator.
    # Finally, this should only work for one single operator, which becomes
    # `selected_operator`.
    kb = set(state.literals) | {action}

    selected_operator = None
    assignment = None
    action_literals = []
    for operator in operators:
        if isinstance(operator.preconds, Literal):
            conds = [operator.preconds]
        else:
            conds = operator.preconds.literals
        # Check whether action is in the preconditions
        action_literal = None
        for lit in conds:
            if lit.predicate == action.predicate:
                action_literal = lit
                break
        action_literals.append(action_literal)
        if action_literal is None:
            continue
        # For proving, consider action variable first
        action_variables = action_literal.variables
        variable_sort_fn = lambda v : (not v in action_variables, v)
        # print(kb)
        # print(conds)
        assignments = find_satisfying_assignments(
            kb, conds,
            variable_sort_fn=variable_sort_fn,
            type_to_parent_types={obj.var_type: {obj.var_type} for obj in state.objects})
        # print(assignments)
        num_assignments = len(assignments)
        if num_assignments > 0:
            # assert num_assignments == 1, "Nondeterministic envs not supported"
            # assignment = assignments[0]
            selected_operator = operator
            break
    # print(selected_operator, assignment)
    # Step 2: Gather up the lifted effects (handles stochastic effects by
    # using the max likelihood one).
    if selected_operator is None:
        return []
    lifted_effects = selected_operator.effects
    # determinized_lifted_effects = []
    # # print(lifted_effects)
    # for lifted_effect in lifted_effects.literals:
    #     # If this effect is probabilistic...
    #     if isinstance(lifted_effect, ProbabilisticEffect):
    #         sampled_effect = lifted_effect.max()  # MOST LIKELY effects
    #         if isinstance(sampled_effect, LiteralConjunction):
    #             for lit in sampled_effect.literals:
    #                 determinized_lifted_effects.append(lit)
    #         else:
    #             determinized_lifted_effects.append(sampled_effect)
    #     else:
    #         determinized_lifted_effects.append(lifted_effect)
    # print(f"step 2 determinized_lifted_effects = {determinized_lifted_effects}")
    """
    probability 0.1
    (not (holding ?x))
    (clear ?x)
    (handempty ?robot)
    (not (handfull ?robot))
    (ontable ?x)
    (table-destroyed)

    probability 0.9
    (not (holding ?x))
    (clear ?x)
    (handempty ?robot)
    (not (handfull ?robot))
    (ontable ?x)
    NO CHANGE

    """
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

    assert np.sum(probs) == 1
    all_prob_effects = list(zip(all_effects, probs)) 
    states = []
    for assignment in assignments:
        for prob_effect_literals, prob in all_prob_effects:
            # Step 3: Apply the effects using the found assignment.
            new_literals = set(state.literals)
            for lifted_effect in prob_effect_literals:
                if lifted_effect == "NOCHANGE":
                    continue
                effect = ground_literal(lifted_effect, assignment)
                # Negative effect
                if effect.is_anti:
                    literal = effect.inverted_anti
                    if literal in new_literals:
                        new_literals.remove(literal)
            for lifted_effect in prob_effect_literals:
                if lifted_effect == "NOCHANGE":
                    continue
                effect = ground_literal(lifted_effect, assignment)
                if not effect.is_anti:
                    new_literals.add(effect)
            states.append((prob/len(assignments), state.with_literals(new_literals)))
    # if len(states) > 1 or len(probs) > 1:
    #     import ipdb; ipdb.set_trace()
    return states

def run_plan(state, plan, operators, random_state):
    plan_queue = [(state, 0)]
    if len(plan) == 0:
        return state
    while len(plan_queue) > 0:
        current_state, idx = plan_queue.pop()
        # print(current_state, idx)
        assert idx < len(plan)
        action = plan[idx]
        output = get_next_state(current_state, action, operators)
        if len(output) > 0:
            probs, next_states = zip(*output)
            sorted_prob_indices = np.argsort(probs)
        else:
            next_states = []
            sorted_prob_indices = []
        next_states_with_idx = [(next_states[j], idx+1) for j in sorted_prob_indices]
        plan_queue = plan_queue + next_states_with_idx
        if (idx == len(plan) - 1) and (len(next_states_with_idx) > 0):
            current_state = next_states_with_idx[random_state.choice(len(next_states_with_idx))][0]
            # import ipdb; ipdb.set_trace()
            return current_state
    raise Exception("could not complete plan")

# _get_next_state(state, self._plan[0], self._learned_operators)