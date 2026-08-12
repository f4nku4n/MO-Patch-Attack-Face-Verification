import os
import json
import argparse
import pickle as p

from utils.common import set_seed, NumpyEncoder, checkSameConfigs

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
    parser.add_argument('--fitness_type', type=str, default='normal', choices=['normal', 'adaptive'],
                        help="The fitness function")

    parser.add_argument('--pop_size', type=int, default=80, help="Population size")
    parser.add_argument('--patch_size', type=int, default=20, help="Size of the patch")

    parser.add_argument('--max_iter', type=int, default=10000, help="Maximum number of generations")
    parser.add_argument('--max_query', type=int, default=10000, help="Maximum number of evaluations")

    parser.add_argument('--prob_mutate_patch', type=float, default=0.3, help="Mutation probability for the content")
    parser.add_argument('--prob_mutate_location', type=float, default=0.5, help="Mutation probability for the location")

    parser.add_argument('--attack_w', type=float, default=0.5, help="Weight for attack fitness")
    parser.add_argument('--recons_w', type=float, default=0.5, help="Weight for reconstruction fitness")
    parser.add_argument('--tournament_size', type=int, default=4, help="Tournament size for selection")

    parser.add_argument('--early_stop', action='store_true', help='Early stop if all individual are the same')

    parser.add_argument('--terminated_condition', type=str, default='generation')
    parser.add_argument('--problem_type', type=str, default='maximizing', choices=['maximizing', 'minimizing'])

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--victim_model_name', type=str, default='resnet',
                        choices=['vggface', 'webface', 'arcface', 'cosface'],
                        help='pretrained victim model')
    parser.add_argument('--n_tested_imgs', type=int, default=100, help="the number of tested images")

    parser.add_argument('--pair_path', type=str, default='lfw_preprocess/pairs.txt')
    parser.add_argument('--img_dir', type=str, default='lfw_preprocess/lfw_crop_margin_5')
    parser.add_argument('--model_dir', type=str, default='./pretrained_model')
    parser.add_argument('--mask_dir', type=str, default='./mask')
    parser.add_argument('--exp_dir', type=str, default='./exp_results')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Save configurations
    config = {
        'baseline': args.baseline,
        'crossover_type': args.crossover_type,
        'fitness_type': args.fitness_type,
        'pop_size': args.pop_size,
        'patch_size': args.patch_size,
        'max_iter': args.max_iter,
        'max_query': args.max_query,
        'prob_mutate_patch': args.prob_mutate_patch,
        'prob_mutate_location': args.prob_mutate_location,
        'attack_w': args.attack_w,
        'recons_w': args.recons_w,
        'tournament_size': args.tournament_size,
        'early_stop': args.early_stop,
        'terminated_condition': args.terminated_condition,
        'problem_type': args.problem_type,
        'seed': args.seed,
        'n_tested_imgs': args.n_tested_imgs,
        'victim_model': args.victim_model_name,
        'exp_dir': args.exp_dir,
    }
    # Create folder 'exp_results'. If it is existed, pass
    exp_dir = args.exp_dir
    if args.terminated_condition == 'generation':
        exp_dir = f'{exp_dir}/{args.baseline}_{args.fitness_type}_maxGen-{args.max_iter}_VictimModel-{args.victim_model_name}/Seed{args.seed}'
    else:
        exp_dir = f'{exp_dir}/{args.baseline}_{args.fitness_type}_maxQuery-{args.max_query}_VictimModel-{args.victim_model_name}/Seed{args.seed}'
    os.makedirs(exp_dir, exist_ok=True)

    continue_exp = False
    if os.path.isfile(f'{exp_dir}/configs.json'):
        prev_config = json.load(open(f'{exp_dir}/configs.json'))
        continue_exp = checkSameConfigs(config, prev_config)

    if not continue_exp:
        with open(f'{exp_dir}/configs.json', "w") as file:
            json.dump(config, file, indent=4, cls=NumpyEncoder)

    exp_img_dir = f"{exp_dir}/images"
    exp_log_dir = f"{exp_dir}/logs"

    os.makedirs(exp_img_dir, exist_ok=True)
    os.makedirs(exp_log_dir, exist_ok=True)

    # Load pre-trained model for the face verification task
    MODEL = get_model(model_name=args.victim_model_name, model_dir=args.model_dir)
    print('Load Pre-trained model - Done!')

    # Load data
    DATA = LFW(IMG_DIR=args.img_dir, MASK_DIR=args.mask_dir, PAIR_PATH=args.pair_path, transform=None)
    print('Load Data - Done!')

    toTensor = transforms.ToTensor()

    random_seed = args.seed
    n_tested_imgs = args.n_tested_imgs
    img_height, img_width = 160, 160  # 160 seem like the required shape of the pre-trained model VGGFace

    success_list = []

    for i in range(n_tested_imgs):
        if os.path.isfile(f'{exp_log_dir}/{i}.p') and continue_exp:
            continue
        set_seed(random_seed)

        img1, img2, label = DATA[i]
        img1, img2 = img1.resize((img_height, img_width)), img2.resize((img_height, img_width))
        img1_torch, img2_torch = toTensor(img1), toTensor(img2)

        population = Population(pop_size=args.pop_size, patch_size=args.patch_size, img_shape=(img_height, img_width),
                                prob_mutate_location=args.prob_mutate_location,
                                prob_mutate_patch=args.prob_mutate_patch)

        fitness = Fitness(img1=img1_torch, img2=img2_torch, model=MODEL, label=label,
                          recons_w=args.recons_w, attack_w=args.attack_w, fitness_type=args.fitness_type,
                          multi_objective=(args.baseline == 'NSGAII'))
        if not fitness.init_self_check():
            print(f'Image #{i + 1}/{n_tested_imgs} - Model gives wrong prediction at initialization!')
            continue

        best_psnr_success, best_ind_success = None, None

        if args.baseline in ['GA', 'GA_rules']:
            using_rules = (args.baseline == 'GA_rules')
            algo = GA(max_iter=args.max_iter, max_query=args.max_query,
                      population=population, fitness=fitness, tournament_size=args.tournament_size,
                      crossover_type=args.crossover_type, terminated_condition=args.terminated_condition,
                      problem_type=args.problem_type, early_stop=args.early_stop, using_rules=using_rules)

        elif args.baseline == "NSGAII":
            algo = NSGAII(max_iter=args.max_iter, max_query=args.max_query,
                          population=population, fitness=fitness, crossover_type=args.crossover_type,
                          terminated_condition=args.terminated_condition, problem_type=args.problem_type,
                          early_stop=args.early_stop)
        else:
            # Give an error if the user inputs the wrong value for the 'baseline' hyperparameter
            raise ValueError

        best_patch = algo.solve()
        patch, loc = best_patch.patch, best_patch.location
        adv_img = fitness.apply_patch_to_image(patch, loc)

        adv_score, psnr_score = best_patch.adv_score.item(), best_patch.psnr_score.item()
        success_attack = (adv_score >= 0)
        success_list.append(success_attack)
        print(f'Image #{i + 1}/{n_tested_imgs}')
        print(f"Adv Score: {adv_score:.4f}")
        print(f"PSNR Score: {psnr_score:.4f}")
        print('Success Attack:', success_attack)
        print('-' * 20)

        # Save_image
        save_image(adv_img, f"{exp_img_dir}/{i}_Success-{success_attack}_AdvScore-{adv_score:.2f}_PSNRScore-{psnr_score:.2f}.png")

        # Save results
        results = {
            "loc": loc,
            "patch": patch.cpu().detach().numpy(),
            "cls_img": img1_torch.cpu().detach().numpy(),
            "adv_img": adv_img.cpu().detach().numpy(),
            "success_attack": success_attack,
            "adv_score": adv_score,
            "psnr_score": psnr_score,
            "list_adv_psnr_scores": algo.history,
        }
        p.dump(results, open(f'{exp_log_dir}/{i}.p', 'wb'))

        # Save final population (for Multi-objective Evolutionary Algorithms)
        if args.baseline in ['NSGAII']:
            exp_pop_dir = f"{exp_dir}/final_pop"
            os.makedirs(exp_pop_dir, exist_ok=True)
            pop = []
            for idv in algo.pop.P:
                info = {'patch': idv.patch.cpu().detach().numpy(), 'loc': idv.location,
                        'adv_score': idv.adv_score.item(), 'psnr_score': idv.psnr_score.item()}
                pop.append(info)
            p.dump(pop, open(f'{exp_pop_dir}/{i}_pop.p', 'wb'))

    print(f"Success rate: {sum(success_list) / len(success_list)}")
