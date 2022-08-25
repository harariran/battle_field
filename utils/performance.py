# For data and graphs
class Dashboard:
    def __init__(self, env):
        pass


# Defining an objective function for soft comparison between joint plans
def objective(joint_plan, obs_seq, total_rewards):
    return colored_total_rewards(total_rewards)


# Summing rewards for each side
def colored_total_rewards(total_rewards):
    red_score = 0
    blue_score = 0
    for reward_dict in total_rewards:
        red_score += round(sum([reward for (agent_id, reward) in reward_dict.items() if 'red' in agent_id]), 2)
        blue_score += round(sum([reward for (agent_id, reward) in reward_dict.items() if 'blue' in agent_id]), 2)
    return {'blue': blue_score, 'red': red_score}


# Defining hard constraints between pair of plans and their corresponding observations
def forbidden_plans(obs_seq, plan_a, agent_id_a, plan_b, agent_id_b, est_poses):
    return plans_collide(obs_seq, agent_id_a, plan_a, agent_id_b, plan_b, est_poses)


# Check if two plans (by their simulated observations) take the same spot in the same time
def plans_collide(obs_seq, agent_id_a, plan_a, agent_id_b, plan_b, est_poses):
    a_pos_seq = est_poses[agent_id_a]
    b_pos_seq = est_poses[agent_id_b]
    collisions = [a_pos_seq[i] == b_pos_seq[i] for i in range(len(obs_seq))]
    return any(collisions)

