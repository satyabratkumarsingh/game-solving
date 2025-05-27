import json
import numpy as np

class Analysis:
    def __init__(self, game_class):
        self.game_class = game_class  # The specific game you are analyzing (e.g., KuhnPoker)
        self.game_data = []  # To hold data from the game rounds
        self.avg_data = {}  # To hold average game results

    def load_game_data(self, files):
        """
        Loads data from the given files (the game result files).
        """
        for file in files:
            with open(file, 'r') as f:
                game_results = json.load(f)
                self.game_data.append(game_results)

    def add_avg(self, files, model_label):
        """
        This function will process the game data from the files, calculate averages, 
        and store the results for analysis.
        """
        self.load_game_data(files)  # Load game data from files

        # Example of processing the data and calculating averages
        scores = []
        for round_data in self.game_data:
            # Assuming the game data contains a list of "scores" for each round
            round_scores = round_data.get('scores', [])
            scores.extend(round_scores)  # Collect all the scores across all rounds

        avg_score = np.mean(scores)  # Calculate the average score
        self.avg_data[model_label] = avg_score  # Store the average score with the model label

    def display_avg_data(self):
        """
        Displays the average game results for all models.
        """
        print("Average Game Results:")
        for model_label, avg_score in self.avg_data.items():
            print(f"Model: {model_label}, Average Score: {avg_score}")

    def plot_results(self):
        """
        This method would plot the average results for all models.
        """
        import matplotlib.pyplot as plt
        
        model_labels = list(self.avg_data.keys())
        avg_scores = list(self.avg_data.values())

        plt.bar(model_labels, avg_scores)
        plt.xlabel('Model')
        plt.ylabel('Average Score')
        plt.title(f"Average Scores for {self.game_class.__name__}")
        plt.xticks(rotation=45)
        plt.show()
