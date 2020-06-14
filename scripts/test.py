import yaml
import argparse
import random
from utils.model import ImageTranslationBranch, NeuralRankingModule
from utils.base_util import *
import time


def get_loss(input_thre):
    # ------sc------
    sc_thre_image = input_thre[[0]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hw, hw).clamp(0.0, 5.0)
    thek = scf_tensor * fs_tensor * fs_tensor - sc_thre_image * openglnv_tensor
    thek = torch.max(thek, comparison_tensor)
    sc_mask = thek / (openglnv_tensor * 0.03 + thek + 1e-12)
    sc_mask = sc_mask.clamp(0.0, 1.0)

    # ------ridge------
    r_thre_image = input_thre[[1]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hw, hw).clamp(0.0, 5.0)
    r_mask = 1.0 - r_thre_image / (fs_tensor * rf_tensor + 1e-12)
    r_mask = r_mask.clamp(0.0, 1.0)

    # ------valley------
    v_thre_image = input_thre[[2]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hw, hw).clamp(0.0, 5.0)
    v_mask = 1.0 - v_thre_image / (fs_tensor * vf_tensor + 1e-12)
    v_mask = v_mask.clamp(0.0, 1.0)

    # ------apr------
    apr_thre_image = input_thre[[3]].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hw, hw).clamp(0.0, 5.0)
    apr_mask = 1.0 - apr_thre_image / (fs_tensor * fs_tensor * aprf_tensor + 1e-12)
    apr_mask = apr_mask.clamp(0.0, 1.0)

    # ------max pooling------

    final_pr = torch.cat((sc_mask * sc_tensor, r_mask * ridge_tensor, v_mask * valley_tensor, apr_mask * apr_tensor, IT_output_with_base), dim=0)

    final_pr, _ = torch.max(final_pr, dim=0, keepdim=True)
    final_pr = final_pr.clamp(0.0, 1.0)
    input_f = torch.cat((final_pr, nv_tensor, depth_tensor), dim=1)
    score = NRM(input_f)
    return score, final_pr


parser = argparse.ArgumentParser()
parser.add_argument('-model_name', type=str, required=True)
parser.add_argument('-config_file', type=str, default='configs/default_config.yml')
parser.add_argument('-save_name', type=str, required=True)
args = parser.parse_args()

conf = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
start_time = time.time()
torch.manual_seed(conf['seed_num'])
torch.cuda.manual_seed(conf['seed_num'])
torch.cuda.manual_seed_all(conf['seed_num'])  # if you are using multi-GPU.
np.random.seed(conf['seed_num'])  # Numpy module.
random.seed(conf['seed_num'])  # Python random module.
torch.manual_seed(conf['seed_num'])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

folder_pos = os.path.join(conf['data_folder'], args.model_name)
with torch.no_grad():
    IT_branch = ImageTranslationBranch(7, 1, ngf=64, n_downsampling=4, n_blocks=9)
    IT_branch.cuda()
    IT_branch.eval()
    IT_branch.load_state_dict(torch.load(conf['ITB_path']))
    IT_input = fetch_IT_input(folder_pos)
    IT_output_probability = IT_branch(IT_input)
    IT_output_np = np.squeeze(IT_output_probability.data.cpu().float().numpy())
    IT_output_with_base = ridge_detection(IT_output_np, folder_pos, conf)

hw = 1024
NRM = NeuralRankingModule(num_classes=1, avg_size=16, have_tanh=False)
NRM.cuda()
NRM.eval()
NRM.load_state_dict(torch.load(conf['NRM_path']))
for param in NRM.parameters():
    param.requires_grad = False

nv_tensor, depth_tensor, sc_tensor, openglnv_tensor, scf_tensor, fs_tensor, comparison_tensor, ridge_tensor, rf_tensor, valley_tensor, vf_tensor, apr_tensor, aprf_tensor = fetch_G_input(folder_pos)

init_np = np.random.uniform(size=(conf['initialization_num'], 4))
init_list = init_np.tolist()

optimized_list = []  # store optimized para for each initialization seed


for init_idx, init_para in enumerate(init_list):  # loop over each initialization

    good_loss = 2e10
    good_para = None

    thre_tensor = torch.tensor(init_para).cuda().requires_grad_()
    optimizer = torch.optim.LBFGS([thre_tensor], lr=conf['lr'])
    already_skip_num = 0
    for iter_now in range(conf['num_iteration']):

        print(init_idx, iter_now)
        skip_indicator = True

        def closure():
            global good_loss
            global skip_indicator
            global good_para
            optimizer.zero_grad()
            loss, the_pr = get_loss(thre_tensor)
            loss.backward()
            if loss.item() < good_loss:
                skip_indicator = False
                good_loss = loss.item()
                good_para = list(thre_tensor.detach().cpu().numpy())  # record better para
            return loss


        optimizer.step(closure)

        if skip_indicator:
            already_skip_num += 1
        if already_skip_num >= conf['skip_threshold']:
            break

    optimized_list.append(good_para)

# ---- save results ----
loss_list = []
pr_list = []
for seed_idx, _ in enumerate(init_list):
    thre_tensor = torch.tensor(optimized_list[seed_idx]).cuda().requires_grad_()
    loss, the_pr = get_loss(thre_tensor)
    loss_list.append(loss.item())
    pr_list.append(the_pr)

best_seed_idx = loss_list.index(min(loss_list))
show_tensor(pr_list[best_seed_idx], args.save_name, crop=True)
total_min = (time.time() - start_time) / 60.0
print(total_min)