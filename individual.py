import torch
import random

class Individual:
    def __init__(self, patch_size, img_shape, prob_mutate_patch=0.3, prob_mutate_location=0.5):
        """
        Initialize an individual with a random patch and location.
        """
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.patch, self.location = None, None
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location

        # Two attributes for the NSGA-II algorithm
        self.rank, self.crowding = None, None

        # Two metrics for evaluating the quality of the patch
        self.psnr_score, self.adv_score = None, None
        self.F = None

        self._random_patch()
        self._random_location()

        self.add_data = {}

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.add_data[key] = value

    def _random_location(self):
        """
        Generates a random location (x_min, x_max, y_min, y_max) within image bounds.
        """
        x_min = random.randint(0, self.img_shape[0] - self.patch_size)
        y_min = random.randint(0, self.img_shape[1] - self.patch_size)
        x_max, y_max = x_min + self.patch_size, y_min + self.patch_size
        
        self.location = (x_min, x_max, y_min, y_max)
    
    def _random_patch(self):
        """
        Generates a random patch of shape (3, patch_size, patch_size).
        """
        self.patch = torch.rand(3, self.patch_size, self.patch_size).cuda()
    
    def mutate(self):
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  # Add rectangle
            self._add_rectangle()
        
        if random.random() < self.prob_mutate_location:
            self._random_location()
 
    def mutate_location(self):
        """
        Apply a mutation location to the individual
        """

        self._random_location()
    
    def mutate_content(self):
        """
        Apply a mutation to the individual: add a rectangle or circle shape to the patch.
        """
        if random.random() < self.prob_mutate_patch:  
            self._add_rectangle()
    
    def _add_rectangle(self):
        """
        Add a rectangle to the patch.
        """
        x_min = random.randint(0, self.patch_size - 1)
        y_min = random.randint(0, self.patch_size - 1)
        width = random.randint(2, 5)
        color = torch.rand(3).cuda()  # Random RGB color

        self.patch[:, x_min: x_min + width, y_min: y_min + width] = color.unsqueeze(1).unsqueeze(2)

    def crossover_UX(self, parent2):
        """
        Perform crossover with another individual to produce two offspring.

        :param parent2: Another Individual object.
        :return: Two new Individual objects.
        """
        offspring1_patch = self.patch.clone()
        offspring2_patch = parent2.patch.clone()

        offspring1 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)
        offspring2 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)

        cut_point = random.randint(0, self.patch_size)
        offspring1_patch[:, :cut_point, :] = parent2.patch[:, :cut_point, :]
        offspring2_patch[:, :cut_point, :] = self.patch[:, :cut_point, :]

        if random.random() < 0.05:
            offspring1.location = parent2.location
            offspring2.location = self.location
      
        offspring1.patch = offspring1_patch
        offspring2.patch = offspring2_patch

        return offspring1, offspring2
    
    def crossover_blended(self, parent2, alpha=0.5):
        """
        using crossover_blended
        o1 = alpha * p1 + (1 - alpha) * p2
        o2 = alpha *p2 + (1 - alpha) * p1
        """    
        offspring1_patch = alpha * self.patch + (1 - alpha) * parent2.patch
        offspring2_patch = alpha * parent2.patch + (1 - alpha) * self.patch

        offspring1 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)
        offspring2 = Individual(self.patch_size, self.img_shape, self.prob_mutate_patch, self.prob_mutate_location)

        if random.random() < 0.05:
            offspring1.location = parent2.location
            offspring2.location = self.location
        
        offspring1.patch = offspring1_patch
        offspring2.patch = offspring2_patch

        return offspring1, offspring2
