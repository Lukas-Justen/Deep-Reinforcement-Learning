class EnvironmentAgent:
  
    def __init__(self, environment, qlearner, verbose = True, weights=None):
        '''
        Initialize the Framework for the Training Environment. We can set training
        relevant parameters right here and take the environment/agent as parameters.
        '''
        self.environment = environment
        self.episodes = 10000
        self.steps = 5000
        self.average_steps = 0
        self.average_reward = 0
        self.logging_steps = 100
        self.qlearner = qlearner
        self.history_episodes = []
        self.history_steps = []
        self.history_rewards = []
        self.verbose = verbose
        if weights:
            self.qlearner.load_weights(weights)

    
    def train_agent(self, train=True):
        '''
        Sets up the training loop for multiple epsidoes & steps in each episode. 
        Here we never need to change anything to test our different agents.
        '''
        for e in range(self.episodes):
            observation, done, t = self.qlearner.reset()

            while t < self.steps and not done:
                action = self.qlearner.take_action(observation, train)
                if not train:
                    self.environment.render()
                next_observation, reward, done, info = self.qlearner.take_step(action)
                self.average_reward += reward
                observation = self.qlearner.remember(observation, action, reward, next_observation, done)
                t += 1
            self.record_training(t, e)
            if train:
                self.qlearner.train_model(e, done)
        
        
    def record_training(self, steps, episode):
        '''
        Prints training relevant information on the command line & stores the values
        in lists so that we can save them on disk to create visualizations.
        '''
        self.average_steps += steps
        if (episode + 1) % self.logging_steps == 0:
            a_steps = self.average_steps/self.logging_steps
            a_reward = self.average_reward/self.logging_steps
            if self.verbose:
                print(episode + 1, self.episodes, a_steps, a_reward)
            self.history_episodes.append(episode + 1)
            self.history_steps.append(a_steps)
            self.history_rewards.append(a_reward)
            self.average_steps = 0
            self.average_reward = 0      
    
    
    def close_environment(self):
        '''
        Closes the environment that we are using for training.
        '''
        if type(self.qlearner).__name__ == 'PolicyGradient':
            self.qlearner.close_policy()
        self.environment.close()
    
    
    def save_models(self, prefix, folder_id):
        '''
        Saves given qlearner model on disk by appending given prefix to file name.
        '''
        self.qlearner.save_models(prefix, folder_id)
    
    
    def get_training_history(self):
        '''
        Returns the average steps and average reward per episode step.
        '''
        history_df = pd.DataFrame()
        history_df["Episodes"] = self.history_episodes
        history_df["Steps"] = self.history_steps
        history_df["Rewards"] = self.history_rewards
        return history_df
  
  
    def save_history(self, prefix, folder_id):
        '''
        Loads the history of the steps and rewards to the Google drive folder.

        :param prefix: The prefix for the file that will be saved on disk.
        '''
        history_df = self.get_training_history()
        history_df.to_csv(prefix + "_history.csv", index = False, header = True)