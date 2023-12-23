# intern_rl
an end-to-end optimal trade execution framework based on Proximal Policy Optimization

## Run

* First run the `envs/train_with_factors.py`: build the model, learn and evaluate

* Parameter need to change

  `model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.002, n_steps=20*20, gamma=1, batch_size=20, tensorboard_log=log_path)`

  * `MlpPolicy`: feature extractor.    `MlpPolicy`, `CnnPolicy`(need picture-format data), `LstmPolicy`

    

  `model.learn()`

  * `total_timesteps`: 3000000

  

* Then run the `envs/test_with_factors.py`, test the model saved in `train_with_factor`



## The basic training process of PPO

input observation and extract features through Lstm/Cnn/Flatten 

construct an MLP that receives the output from a previous features extractor (i.e. a CNN/Lstm) or directly the observations (if no features extractor is applied) as an input    `pop_new/torch_layers/MlpExtractor`

output a latent representation for the policy and a value network

get the actions, value    `pop_new/NewPolicy/ActorCriticPolicy`




## The changing part ppo_new

`NewPolicy`: add `ActorCriticLstmPolicy`

`torch_layer`: add `LstmExtractor`

> `LstmExtractor`: input the observation, then through `hidden_layer` and `self.linear`, `flatten[in order to get one dimension action] ` then get the output. 
>
> Parameter:
>
> * n_input_size: the dimension of the data input
> * Features_dim=hidden_size: should be larger than n_input_size
>
> In sum, here the `Features_dim` and `activation function` can defined freely.





