# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
# from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from ppo_new.NewPolicy import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy
from NewPolicy import ActorCriticLstmPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
LstmPolicy = ActorCriticLstmPolicy
