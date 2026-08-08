from individual import Individual

class Population:
    def __init__(self, pop_size, patch_size, img_shape, prob_mutate_patch, prob_mutate_location):
        self.pop_size = pop_size
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location
        
        self._create_population(patch_size, img_shape, prob_mutate_patch, prob_mutate_location)
        
    def _create_population(self, patch_size, img_shape, prob_mutate_patch, prob_mutate_location):
        self.P = [Individual(patch_size, img_shape, prob_mutate_patch, prob_mutate_location) for _ in range(self.pop_size)]
