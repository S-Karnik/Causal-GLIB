"""Curiosity module that samples previously unachieved goals and plans to
achieve them with the current operators.
"""
from copy import deepcopy
import numpy as np
import os
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from settings import AgentConfig as ac
from curiosity_modules.GLIB_grounded import GLIBG1CuriosityModule
from run_gym_iter import get_next_state


class GLIBIntResetEvalCuriosityModule(GLIBG1CuriosityModule):
    _k = None # Must be set by subclasses
    """Curiosity module that samples a completely random literal and plans to
    achieve it with the current operators.
    """
    def _initialize(self):
        self._num_steps = 0
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "glibg1_int_reset_eval"
        self._static_preds = self._compute_static_preds()
        # Keep track of the number of times that we follow a plan
        self.line_stats = []

    def _get_action(self, state):
        if self._unseen_lits_acts is None:
            self._recompute_unseen_lits_acts(state)
        action = self._get_action_goal_babble(state)
        for lit in state:  # update novelty
            if (lit, action) in self._unseen_lits_acts:
                self._unseen_lits_acts.remove((lit, action))
        return action

    def _get_action_goal_babble(self, state):
        last_state = self._last_state
        self._last_state = state

        # Continue executing plan?
        if self._plan and (last_state != state):
            self.line_stats.append(1)
            # print("CONTINUING PLAN")
            return self._plan.pop(0)

        # Try to sample a goal for which we can find a plan
        sampling_attempts = planning_attempts = 0
        best_plan = []
        best_goal = None
        best_last_sampled_action = None
        best_plan_reward = -1
        found_at_least_one_plan = False
        goals = self._sample_goal_k(state, ac.max_sampling_tries)
        if goals is not None:
            goals, last_sampled_actions = goals

        while (goals is not None and \
            sampling_attempts < len(goals) and \
            planning_attempts < ac.max_planning_tries):

            goal = goals[sampling_attempts]
            self._last_sampled_action = last_sampled_actions[sampling_attempts]
            sampling_attempts += 1

            # print("trying goal:",goal)

            # Create a pddl problem file with the goal and current state
            problem_fname = self._create_problem_pddl(
                state, goal, prefix=self._name)

            # Get a plan
            try:
                self._plan = self._planning_module.get_plan(
                    problem_fname, use_cache=False)
                old_learned_ops = deepcopy(self._learned_operators)

                plan_state = state
                # Do hypothetical update.
                plan_queue = [(state, 0)]
                assigned_next_states = [None for i in range(len(self._plan))]
                idx = 0
                while len(plan_queue) > 0:
                    plan_state, idx = plan_queue.pop()
                    if idx > 0:
                        assigned_next_states[idx-1] = plan_state
                    if idx == len(self._plan):
                        break
                    action = self._plan[idx]
                    next_states = [(s, idx+1) for s in get_next_state(plan_state, action, self._learned_operators)]
                    plan_queue = plan_queue + next_states

                plan_state = state
                plan_with_states = []
                for i, action in enumerate(self._plan):
                    plan_with_states.append((plan_state, action, assigned_next_states[i]))
                    plan_state = assigned_next_states[i]

                for plan_state, action, plan_next_state in plan_with_states:
                    effects = self._compute_effects(plan_state, plan_next_state)
                    self._operator_learning_module.observe(plan_state, action, effects)

                state_after_plan = state if len(plan_with_states) == 0 else plan_with_states[-1][-1]

                # NEVER DO equal to!!!!
                # self._learned_operators = (compute hypothetical operators under this plan)
                # alternatively, just have a helper method "update_learned_operators_in_all_modules(learned_operators)"
                all_rewards = self._get_all_current_rewards_val_tasks(state_after_plan, self._name, self._rand_state, reset_eval=True)  # uses your updated self._learned_operators
                avg_reward = np.mean(all_rewards)
                if avg_reward > best_plan_reward:
                    best_last_sampled_action = self._last_sampled_action
                    best_plan_reward = avg_reward
                    found_at_least_one_plan = True
                    best_goal = goal
                    best_plan = self._plan
                self._learned_operators.clear()
                for op in old_learned_ops:
                    self._learned_operators.add(op)
            except NoPlanFoundException:
                os.remove(problem_fname)
                continue
            except PlannerTimeoutException:
                os.remove(problem_fname)
                continue
            os.remove(problem_fname)
            planning_attempts += 1
            self._plan = []

        if found_at_least_one_plan:
            self._last_sampled_action = best_last_sampled_action
            self._plan = self._finish_plan(best_plan)
            print("\tGOAL:", best_goal)
            print("\tPLAN:", self._plan)
            # import ipdb; ipdb.set_trace()
            # Take the first step in the plan
            self.line_stats.append(1)
            return self._plan.pop(0)

        # No plan found within budget; take a random action
        # print("falling back to random")
        return self._get_fallback_action(state)

class GLIBG1IntResetEvalCuriosityModule(GLIBIntResetEvalCuriosityModule):
    _k = 1
