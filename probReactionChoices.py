import random
import numpy as np
from scipy import special

class probReactionChoices:
    texts = []
    frequencies = []
    probabilities = []

    def __init__(self, init_texts):
        self.texts = init_texts
        self.frequencies = [0 for i in self.texts]
        self.probabilities = [1./len(self.texts) for i in self.texts]

    def getRandomTextChoice(self):
        choice_idx = np.random.choice(range(len(self.texts)), p=self.probabilities)
        self.frequencies[choice_idx] -= 1
        self._updateProbabilities()
        return self.texts[choice_idx]

    def _updateProbabilities(self):
        self.probabilities = special.softmax(self.frequencies)
        #print("Probabilities:", self.probabilities)

    def getLeastFreqChoice(self):
        #print("Frequencies:", self.frequencies)
        sort_idx_list = np.argsort(self.frequencies)
        #print("Sort idx:", sort_idx_list)
        choice_idx_freq = self.frequencies[sort_idx_list[0]]
        #print("First elem freq:", choice_idx_freq)
        choices = [sort_idx_list[0]]

        i=1
        while i<len(sort_idx_list) and self.frequencies[sort_idx_list[i]] <= choice_idx_freq:
            choices.append(sort_idx_list[i])
            i+=1
        #print("Choices:", choices)
        choice_idx = random.choice(choices)
        self.frequencies[choice_idx] += 1
        return self.texts[choice_idx]


if __name__ == "__main__":
    reaction_neutral_list = [
        "Got it",
        "Thanks for sharing",
        "Got it! Thanks for sharing",
        "Sure",
        "Noted",
        "I got it",
        "Okay, I'm getting a better idea of your answers",
        "Thank you for you answer"
    ]

    #print("Selected only texts:")
    #print(reaction_neutral_list.text)
    pe = probReactionChoices(reaction_neutral_list)
    for i in range(20):
        print("-> Choice:", pe.getLeastFreqChoice())

    #for i in range(10):
    #    print(np.random.choice(reaction_neutral_list, p=[0.1,0.9]))
    #    #print(random.choice(reaction_neutral_list, [0.5,0.5]))

