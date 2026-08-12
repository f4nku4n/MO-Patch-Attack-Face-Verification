from individual import Individual

class Population:
    def __init__(self, pop_size, patch_size, img_shape, prob_mutate_patch, prob_mutate_location, cover_all_image=False):
        self.pop_size = pop_size
        self.patch_size = patch_size
        self.img_shape = img_shape
        self.prob_mutate_patch = prob_mutate_patch
        self.prob_mutate_location = prob_mutate_location
        
        self._create_population(patch_size, img_shape, prob_mutate_patch, prob_mutate_location, cover_all_image)

    def _get_all_locations(self):
        list_locs = []
        img_h, img_w = self.img_shape
        for i in range(0, img_h, self.patch_size):
            for j in range(0, img_w, self.patch_size):
                list_locs.append((i, i + self.patch_size, j, j + self.patch_size))
        return list_locs

    def _create_population(self, patch_size, img_shape, prob_mutate_patch, prob_mutate_location, cover_all=False):
        self.P = [Individual(patch_size, img_shape, prob_mutate_patch, prob_mutate_location) for _ in range(self.pop_size)]
        if cover_all:
            list_locs = self._get_all_locations()
            i = 0
            for idv in self.P:
                idv.location = list_locs[i]
                i += 1
                if i == len(list_locs):
                    i = 0
