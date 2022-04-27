from .curiosity_base import BaseCuriosityModule
from .oracle_curiosity import OracleCuriosityModule
from .random_actions import RandomCuriosityModule
from .goal_babbling import GoalBabblingCuriosityModule
from .GLIB_grounded_no_intervention_causal import *
from .GLIB_grounded_interventional import *
from .GLIB_grounded_interventional_reset_eval import *
from .GLIB_grounded import *
from .GLIB_lifted import *


def create_curiosity_module(curiosity_module_name, action_space,
                            observation_space, planning_module,
                            learned_operators, operator_learning_module, 
                            domain_name, get_current_rewards_val_env,
                            compute_effects):
    module = None
    if curiosity_module_name == "oracle":
        module = OracleCuriosityModule
    elif curiosity_module_name == "random":
        module = RandomCuriosityModule
    elif curiosity_module_name == "GLIB_G1":
        module = GLIBG1CuriosityModule
    elif curiosity_module_name == "goalbabbling":
        module = GoalBabblingCuriosityModule
    elif curiosity_module_name == "GLIB_G1_no_int_causal":
        module = GLIBG1NoIntCausalCuriosityModule
    elif curiosity_module_name == "GLIB_G1_int":
        module = GLIBG1IntCuriosityModule
    elif curiosity_module_name == "GLIB_G1_int_reset_eval":
        module = GLIBG1IntResetEvalCuriosityModule
    elif curiosity_module_name == "GLIB_L1":
        module = GLIBL1CuriosityModule
    elif curiosity_module_name == "GLIB_L2":
        module = GLIBL2CuriosityModule
    else:
        raise Exception("Unrecognized curiosity module '{}'".format(
            curiosity_module_name))
    return module(action_space, observation_space, planning_module,
                  learned_operators, operator_learning_module, domain_name, get_current_rewards_val_env, compute_effects)
