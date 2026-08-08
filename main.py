import os
import argparse
import pickle as p

from utils.common import set_seed

from fitness import Fitness
from algorithm import GA, NSGAII
from population import Population

from dataset import LFW
from factory import get_model

from torchvision import transforms
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Image Patch Manipulation")
    parser.add_argument('--baseline', type=str, default='GA', choices=['GA', 'GA_rules', 'NSGAII'])
    parser.add_argument('--crossover_type', type=str, default='Blended', choices=['UX', 'Blended'])
    parser.add_argument('--fitness_type', type=str, choices=['normal', 'adaptive'], help="The fitness function")

    parser.add_argument('--pop_size', type=int, default=10, help="Population size")
    parser.add_argument('--patch_size', type=int, default=20, help="Size of the patch")

    parser.add_argument('--max_iter', type=int, default=10000, help="Maximum number of generations")
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")

    parser.add_argument('--prob_mutate_patch', type=float, default=0.1, help="Mutation probability for the content")
    parser.add_argument('--prob_mutate_location', type=float, default=0.1, help="Mutation probability for the location")

    parser.add_argument('--attack_w', type=float, default=0.5, help="Weight for attack fitness")
    parser.add_argument('--recons_w', type=float, default=0.5, help="Weight for reconstruction fitness")
    parser.add_argument('--tournament_size', type=int, default=2, help="Tournament size for selection")

    parser.add_argument('--update_location_interval', type=int, default=200)
    parser.add_argument('--terminated_condition', type=str, default='generation')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")

    parser.add_argument('--pair_path', type=str, default='lfw_preprocess/pairs.txt')
    parser.add_argument('--exp_dir', type=str, default='./exp_results')
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Create folder 'exp_results'. If it is existed, pass
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    exp_dir = f"{exp_dir}/{args.baseline}_{args.fitness_type}"
    exp_img_dir = f"{exp_dir}/images"
    exp_log_dir = f"{exp_dir}/logs"

    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    # Load pre-trained model for the face verification task
    MODEL = get_model(model_name="resnet_vggface", model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')

    # Load data
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=args.pair_path, transform=None)
    print('Load Data - Done!')

    toTensor = transforms.ToTensor()
    success_rate = 0
    results = []

    random_seed = args.seed
    n_tested_imgs = args.n_tested_imgs
    img_height, img_width = 160, 160   # 160 seem like the required shape of the pre-trained model VGGFace

    for i in range(n_tested_imgs):
        set_seed(random_seed)

        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((img_height, img_width)), img2.resize((img_height, img_width))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)
        
        population = Population(pop_size=args.pop_size, patch_size=args.patch_size, img_shape=(img_height, img_width),
                                prob_mutate_location=args.prob_mutate_location, prob_mutate_patch=args.prob_mutate_patch)
        
        fitness = Fitness(patch_size=args.patch_size, img1=img1_torch, img2=img2_torch,
                          model=MODEL, label=label,
                          recons_w=args.recons_w, attack_w=args.attack_w, fitness_type=args.fitness_type)

        best_psnr_success, best_ind_success = None, None

        if args.baseline in ['GA', 'GA_rules']:
            using_rules = (args.baseline == 'GA_rules')
            algo = GA(max_iter=args.max_iter, max_query=args.max_query,
                      population=population, fitness=fitness, tournament_size=args.tournament_size,
                      interval_update=args.update_location_iterval, crossover_type=args.crossover_type,
                      terminated_condition=args.terminated_condition, using_rules=using_rules)
        
        elif args.baseline == "NSGAII":
            algo = NSGAII(max_iter=args.max_iter, max_query=args.max_query,
                          population=population, fitness=fitness, tournament_size=args.tournament_size,
                          interval_update=args.update_location_iterval, crossover_type=args.crossover_type,
                          terminated_condition=args.terminated_condition)
        else:
            # Give an error if the user inputs the wrong value for the 'baseline' hyperparameter
            raise ValueError

        adv_img, adv_score, pnsr_score = algo.solve()

        # Save_image
        save_image(adv_img, os.path.join(exp_img_dir, f"{i}.png"))
        #
        # result = {
        #         "adv_img": adv_img,
        #         "adv_score": adv_score,
        #         "pnsr_score": pnsr_score,
        #         "best_psnr_success": best_psnr_success,
        #         "best_ind_success": best_ind_success,
        #         "log": full_P
        #         }
        
        print("Adv score: ", adv_score)
        if adv_score > 0:
            success_rate += 1
        #
        # output_pickle = os.path.join(exp_log_dir, f'{i}.pkl')
        #
        # with open(output_pickle, 'wb') as f:
        #     p.dump(result, f)
                        
    print(f"Success rate: {success_rate / 100}")
