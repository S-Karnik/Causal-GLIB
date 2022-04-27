from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from run_gym_iter import get_next_state, run_plan
from settings import AgentConfig as ac
from pddlgym.structs import Anti, State
import time
import numpy as np
import random
from pddlgym.parser import PDDLProblemParser

class Agent:
    """An agent interacts with an env, learns PDDL operators, and plans.
    This is a simple wrapper around three modules:
    1. a curiosity module
    2. an operator learning module
    3. a planning module
    The curiosity module selects actions to collect training data.
    The operator learning module learns operators from the training data.
    The planning module plans at test time.
    The planning module (and optionally the curiosity module) use the
    learned operators. The operator learning module contributes to them.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name, val_env):
        self.curiosity_time = 0.0
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name
        self.action_space = action_space
        # The main objective of the agent is to learn good operators
        self.learned_operators = set()
        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.learned_operators, self.domain_name)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._val_env = val_env
        self._val_env_goals = [prob.goal for prob in val_env.problems]

        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.learned_operators,
            self._operator_learning_module, domain_name, self.get_all_current_rewards_val_tasks, self.compute_effects)

    ## Training time methods
    def get_action(self, state):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        start_time = time.time()
        action = self._curiosity_module.get_action(state)
        self.curiosity_time += time.time()-start_time
        return action

    def observe(self, state, action, next_state):
        # Get effects
        effects = self.compute_effects(state, next_state)
        # Add data
        self._operator_learning_module.observe(state, action, effects)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time

    def learn(self):
        # Learn (probably less frequently than observing)
        some_operator_changed = self._operator_learning_module.learn()
        if some_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
            self.curiosity_time += time.time()-start_time
        # import ipdb; ipdb.set_trace()
            # for pred, dt in self._operator_learning_module.learned_dts.items():
            #     print(pred)
            #     print(dt.print_conditionals())
            # print()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_operator_changed

    def reset_episode(self, state):
        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        self.curiosity_time += time.time()-start_time

    @staticmethod
    def compute_effects(state, next_state):
        positive_effects = {e for e in next_state.literals - state.literals}
        negative_effects = {Anti(ne) for ne in state.literals - next_state.literals}
        return positive_effects | negative_effects

    ## Test time methods
    def get_policy(self, problem_fname):
        """Get a plan given the learned operators and a PDDL problem file."""
        return self._planning_module.get_policy(problem_fname)

    def _create_problem_pddl(self, state, goal, prefix):
        fname = "tmp/{}_problem_{}.pddl".format(
            prefix, random.randint(0, 9999999))
        objects = state.objects
        all_action_lits = self.action_space.all_ground_literals(state)
        initial_state = state.literals | all_action_lits
        problem_name = "{}_problem".format(prefix)
        domain_name = self._planning_module.domain_name
        PDDLProblemParser.create_pddl_file(fname, objects, initial_state,
                                           problem_name, domain_name, goal)

        return fname

    def get_all_current_rewards_val_tasks(self, initial_state, sol_name, rand_state, reset_eval=False):
        rewards = []
        for i in range(len(self._val_env.problems)):
            # import ipdb; ipdb.set_trace()
            goal = self._val_env.problems[i].goal
            if reset_eval:
                initial_state = State(frozenset(self._val_env.problems[i].initial_state),
                              frozenset(self._val_env.problems[i].objects),
                              self._val_env.problems[i].goal)
                 
            else:
                goal_blocks = frozenset(sum([l.variables for l in goal.literals], []))
                if not goal_blocks.issubset(initial_state.objects):
                    continue
                initial_state = State(initial_state.literals, initial_state.objects, goal)
            rewards.append(self._get_current_reward_val_tasks(initial_state, goal, sol_name, rand_state))
                
        return rewards

    def _get_current_reward_val_tasks(self, initial_state, goal, sol_name, rand_state):
        problem_fname = self._create_problem_pddl(initial_state, goal, sol_name)
        # print(problem_fname)
        try:
            # import ipdb; ipdb.set_trace()
            plan = self._planning_module.get_plan(problem_fname)
        except (NoPlanFoundException, PlannerTimeoutException):
            # Automatic failure
            return 0
        reward = 0
        goal_literals_set = frozenset(goal.literals)
        plan_state_end = run_plan(initial_state, plan, self.learned_operators, rand_state)
        
        if goal_literals_set.issubset(plan_state_end.literals):
            reward = 1
            
        return reward
