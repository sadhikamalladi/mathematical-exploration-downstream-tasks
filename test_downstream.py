import csv
import pickle as pkl
import os
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import sys
from sklearn.linear_model import LogisticRegressionCV as LR
from argparse import ArgumentParser
import logging
import random
from sklearn.model_selection import StratifiedKFold

D = 768
SOLVER_EXCEPTION_KEYS = [('sst2_fine', 'Phi p_f'),
                         ('sst2_fine', 'subset p_f'),
                         ('sst2_fine', 'class p_f'),
                         ('sst2_fine', 'posneg p_f')]

## NOTE: if you change the prompt, make sure you don't have the prompt embeddings cached
movie_prompt = ' This movie is '
gen_prompt = ' The sentiment is '
news_prompt = ' This article is about '
PROMPTS = {'sst2': movie_prompt,
           'sst2_fine': movie_prompt,
           'imdb': movie_prompt,
           'mpqa': gen_prompt,
           'cr': gen_prompt,
           'agnews': news_prompt}

SENTIMENT_TASKS = ['sst2', 'sst2_fine', 'glue_sst', 'mpqa', 'cr', 'imdb']
SENTIMENT_SUBSET = [':)', ':(', 'great', 'charming', 'flawed', 'classic', 'interesting', 'boring',
                    'sad', 'happy', 'terrible', 'fantastic', 'exciting', 'strong']
SENTIMENT_CLASS_WORDS = [':)', ':(']
SENTIMENT_POSNEG = ['positive', 'negative']

AG_SUBSET = ['world', 'politics', 'sports', 'business', 'science', 'financial', 'market',
             'foreign', 'technology', 'international', 'stock', 'company', 'tech', 'technologies']
AG_CLASS_WORDS = ['foreign', 'sports', 'financial', 'scientific']

##### helper functions
def fit_linear_clf(x, y, cv, rnd_state=0, solver='liblinear', multi_mode='auto'):
    logger.info(f'Training shape: {x.shape}')
    dual_mode = np.less(*x.shape)
    binary = len(np.unique(y)) == 2
    is_small = x.shape[0] < 10000
    min_exp = -6 if is_small else -3
    max_exp = 4 if is_small else 3
    logger.info(f'Using {solver} solver with {multi_mode} multinomial mode')
    clf = LR(fit_intercept=True, Cs=[10 ** i for i in range(min_exp, max_exp)], cv=cv, 
             dual=dual_mode, multi_class=multi_mode, solver=solver,
             n_jobs=-1, random_state=rnd_state, max_iter=200)
    clf.fit(x, y)

    solver_config = {'min_exp': min_exp, 'max_exp': max_exp,
                     'solver': solver, 'multi_mode': multi_mode,
                     'dual': dual_mode, 'rnd_state': rnd_state}
    
    return clf, solver_config

def get_solver(config):
    if config in SOLVER_EXCEPTION_KEYS:
        logger.info(f'For {config[1]} on {config[0]}, we use multinomial mode and sag solver')
        solver = 'sag'
        multi_mode = 'multinomial'
    else:
        solver = 'liblinear'
        multi_mode = 'auto'
    return solver, multi_mode

def find_valid(tokenizer, subset):
    space_before = [' ' + tok for tok in subset]
    space_after = [tok + ' ' for tok in subset]
    total_subset = space_before + subset + space_after 

    valid_tokens = []
    chosen_words = []
    logger.info(f'Checking {len(total_subset)} tokens for validity')
    for word in total_subset:
        toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
        if len(toks) == 1 and toks[0] not in valid_tokens:
            chosen_words.append(word)
            valid_tokens.append(toks[0])
    return valid_tokens, chosen_words

def retrieve_sentiment_subsets(train, test, Phi, hps):
    path = os.path.join(hps.data_dir, 'sentiment_subset.pkl')
    if os.path.exists(path):
        all_info = pkl.load(open(path, 'rb'))
        all_tokens = all_info['tokens']
    else:
        tokenizer = AutoTokenizer.from_pretrained(hps.model)
        subset_tokens, subset_chosen = find_valid(tokenizer, SENTIMENT_SUBSET)
        logger.info(f'Found {len(subset_tokens)} valid tokens out of subset')
        class_tokens, class_chosen = find_valid(tokenizer, SENTIMENT_CLASS_WORDS)
        logger.info(f'Found {len(class_tokens)} valid tokens for class words')
        posneg_tokens, posneg_chosen = find_valid(tokenizer, SENTIMENT_POSNEG)
        logger.info(f'Found {len(posneg_tokens)} valid tokens for positive/negative')
        
        all_info = {}
        all_info['tokens'] = {'subset': subset_tokens, 'class': class_tokens, 'posneg': posneg_tokens}
        all_info['found words'] = {'subset': subset_chosen, 'class': class_chosen, 'posneg': posneg_chosen}
        all_info['attempted words'] = {'subset': SENTIMENT_SUBSET, 'class': SENTIMENT_CLASS_WORDS, 'posneg': SENTIMENT_POSNEG}
        
        pkl.dump(all_info, open(path, 'wb'))

        all_tokens = all_info['tokens']

    tr_sub = {}
    te_sub = {}
    for k in all_tokens.keys():
        train_sub = train.dot(Phi[all_tokens[k]].T)
        tr_sub[k] = train_sub
        test_sub = test.dot(Phi[all_tokens[k]].T)
        te_sub[k] = test_sub

    return tr_sub, te_sub

def retrieve_agnews_subsets(train, test, Phi, hps):
    path = os.path.join(hps.data_dir, 'agnews_subset.pkl')
    if os.path.exists(path):
        all_info = pkl.load(open(path, 'rb'))
        all_tokens = all_info['tokens']
    else:
        tokenizer = AutoTokenizer.from_pretrained(hps.model)
        subset_tokens, subset_chosen = find_valid(tokenizer, AG_SUBSET)
        logger.info(f'Found {len(subset_tokens)} valid tokens out of subset')
        class_tokens, class_chosen = find_valid(tokenizer, AG_CLASS_WORDS)
        logger.info(f'Found {len(class_tokens)} valid tokens for class words')
        
        all_info = {}
        all_info['tokens'] = {'subset': subset_tokens, 'class': class_tokens}
        all_info['found words'] = {'subset': subset_chosen, 'class': class_chosen}
        all_info['attempted words'] = {'subset': AG_SUBSET, 'class': AG_CLASS_WORDS}
        
        pkl.dump(all_info, open(path, 'wb'))

        all_tokens = all_info['tokens']

    tr_sub = {}
    te_sub = {}
    for k in all_tokens.keys():
        train_sub = train.dot(Phi[all_tokens[k]].T)
        tr_sub[k] = train_sub
        test_sub = test.dot(Phi[all_tokens[k]].T)
        te_sub[k] = test_sub

    return tr_sub, te_sub

def load_data(hps):
    task_dir = os.path.join(hps.data_dir, hps.task)
    logger.info(f'Loading data from {task_dir}')
    tr_d = pkl.load(open(os.path.join(task_dir, 'tr_inp.pkl'), 'rb'))
    te_d = pkl.load(open(os.path.join(task_dir, 'te_inp.pkl'), 'rb'))
    tr_y = pkl.load(open(os.path.join(task_dir, 'tr_labels.pkl'), 'rb'))
    te_y = pkl.load(open(os.path.join(task_dir, 'te_labels.pkl'), 'rb'))

    return (tr_d, tr_y), (te_d, te_y)

def load_model_and_tokenizer(hps):
    logger.info(f'Loading model and tokenizer from {hps.model_name}')
    model = AutoModelWithLMHead.from_pretrained(hps.model)
    tokenizer = AutoTokenizer.from_pretrained(hps.model)
    if 'bert' in hps.model:
        assert not hps.untied, 'Not clear what untied model means for BERT'
        tokenizer.eos_token = tokenizer.pad_token
        W = model.cls.predictions.decoder.weight.detach().numpy()
    else:
        if hps.untied:
            W = model.Phi.weight.detach().numpy()
        else:
            W = model.lm_head.weight.detach().numpy()
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        logger.info('We recommend you have a GPU available for generating embeddings for the task')

    return model, tokenizer, W

def probs_means(vecs, w_emb):
    probs = softmax(vecs.dot(w_emb.T), axis=1)
    means = probs.dot(w_emb)
    return probs, means

def load_or_generate_embs(hps): 
    prompt_str = '_prompt' if hps.prompt else ''
    prompt = PROMPTS[hps.task] if hps.prompt else ''
    prompt_msg = f' with prompt {prompt}' if hps.prompt else ' without prompt'
    embs = {}

    Phi = None

    ### f
    path = os.path.join(hps.output_dir, f'{hps.model_name}_f{prompt_str}.pkl')
    if os.path.exists(path):
        logger.info(f'Loading f embeddings{prompt_msg} from {path}')
        all_data = pkl.load(open(path, 'rb'))
        tr_x, tr_y = all_data['tr_x'], all_data['tr_y']
        te_x, te_y = all_data['te_x'], all_data['te_y']
    else:
        model, tokenizer, Phi = load_model_and_tokenizer(hps)
        (tr_d, tr_y), (te_d, te_y) = load_data(hps)
        logger.info(f'{len(tr_d)} train examples, {len(te_d)} test examples')
        tr_x = []
        bs = hps.batch_size
        emb_fn = bert_batch_embed if 'bert' in hps.model else gpt2_batch_embed
        for i in tqdm(range(0, len(tr_d), bs)):
            tr_x.append(emb_fn(tr_d[i:i + bs], model, tokenizer, prompt=prompt))

            if i%5000 == 0:
                np.save(f'{hps.model_name}_fpartial', tr_x)

        te_x = []
        for i in tqdm(range(0, len(te_d), bs)):
            te_x.append(emb_fn(te_d[i:i + bs], model, tokenizer, prompt=prompt))

        tr_x = np.vstack(tr_x)
        te_x = np.vstack(te_x)
        logger.info(f'Saving f embeddings{prompt_msg} to {path}')
        all_data = {'tr_x': tr_x, 'tr_y': tr_y, 'te_x': te_x, 'te_y': te_y}
        pkl.dump(all_data, open(path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
    
    embs['f'] = ((tr_x, tr_y), (te_x, te_y))

    if 'bert' in hps.model:
        return embs

    ### subsets of p
    if hps.task in SENTIMENT_TASKS:
        if Phi is None:
            model, tokenizer, Phi = load_model_and_tokenizer(hps)
        logger.info(f'{hps.task} is a sentiment task so retrieving subset of p')
        tr_subp, te_subp = retrieve_sentiment_subsets(tr_x, te_x, Phi, hps)
        embs['subset p_f'] = ((tr_subp['subset'], tr_y), (te_subp['subset'], te_y))
        embs['class p_f'] = ((tr_subp['class'], tr_y), (te_subp['class'], te_y))
        embs['posneg p_f'] = ((tr_subp['posneg'], tr_y), (te_subp['posneg'], te_y))
    elif hps.task == 'agnews':
        if Phi is None:
            model, tokenizer, Phi = load_model_and_tokenizer(hps)
        logger.info('Retrieving subsets of p for AG News')
        tr_subp, te_subp = retrieve_agnews_subsets(tr_x, te_x, Phi, hps)
        embs['subset p_f'] = ((tr_subp['subset'], tr_y), (te_subp['subset'], te_y))
        embs['class p_f'] = ((tr_subp['class'], tr_y), (te_subp['class'], te_y))
    else:
        logger.info('Will not run subset or class words for this task')

    ### p
    tr_Phip = None
    tr_p = None
    if hps.run_p:
        path = os.path.join(hps.output_dir, f'{hps.model_name}_pf{prompt_str}.pkl')
        if os.path.exists(path):
            logger.info(f'Loading p_f embeddings{prompt_msg} from {path}')
            all_data = pkl.load(open(path, 'rb'))
            tr_p, tr_y = all_data['tr_x'], all_data['tr_y']
            te_p, te_y = all_data['te_x'], all_data['te_y']
        else:
            if Phi is None:
                model, tokenizer, Phi = load_model_and_tokenizer(hps)
            logger.info(f'Computing p_f embeddings{prompt_msg}')
            tr_p, tr_Phip = probs_means(tr_x, Phi)
            te_p, te_Phip = probs_means(te_x, Phi)
            if hps.save_p:
                logger.info(f'Saving p_f embeddings{prompt_msg} to {path}')
                all_data = {'tr_x': tr_p, 'tr_y': tr_y, 'te_x': te_p, 'te_y': te_y}
                # pickle highest protocol argument allows saving files larger than 4GB
                pkl.dump(all_data, open(path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        embs['p_f'] = ((tr_p, tr_y), (te_p, te_y))

    ### Phi p
    path = os.path.join(hps.output_dir, f'{hps.model_name}_Phipf{prompt_str}.pkl')
    if os.path.exists(path):
        logger.info(f'Loading Phi p_f embeddings{prompt_msg} from {path}')
        all_data = pkl.load(open(path, 'rb'))
        tr_Phip, tr_y = all_data['tr_x'], all_data['tr_y']
        te_Phip, te_y = all_data['te_x'], all_data['te_y']
    else:
        # avoid recomputing p_{f(s)} if we have the p_{f(s)} but not Phi p
        # triggered for a lot of large datasets
        if tr_Phip is None and tr_p is not None:
            logger.info(f'Computing Phi p_f embeddings{prompt_msg} from p_f features')
            tr_Phip = tr_p.dot(Phi)
            te_Phip = te_p.dot(Phi)
        elif tr_Phip is None:
            if Phi is None:
                model, tokenizer, Phi = load_model_and_tokenizer(hps)
            logger.info(f'Computing Phi p_f embeddings{prompt_msg} from scratch')
            _, tr_Phip = probs_means(tr_x, Phi)
            _, te_Phip = probs_means(te_x, Phi)
            
        logger.info(f'Saving Phi p_f embeddings{prompt_msg} to {path}')
        all_data = {'tr_x': tr_Phip, 'tr_y': tr_y, 'te_x': te_Phip, 'te_y': te_y}
        pkl.dump(all_data, open(path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
    embs['Phi p_f'] = ((tr_Phip, tr_y), (te_Phip, te_y))

    if hps.run_projections:
        ### A p (random projection)
        path = os.path.join(hps.output_dir, f'{hps.model_name}_Apf{prompt_str}.pkl')
        A_path = os.path.join(hps.output_dir, f'{hps.model_name}_A{prompt_str}.pkl')
        if os.path.exists(path):
            logger.info(f'Loading A p_f embeddings{prompt_msg} from {path}')
            all_data = pkl.load(open(path, 'rb'))
            tr_Ap, tr_y = all_data['tr_x'], all_data['tr_y']
            te_Ap, te_y = all_data['te_x'], all_data['te_y']
        else:
            A = np.random.randn(D, tr_p.shape[1])
            _, _, A = np.linalg.svd(A, full_matrices=False)
            pkl.dump(A, open(A_path, 'wb'))
            tr_Ap = tr_p.dot(A.T)
            te_Ap = te_p.dot(A.T)
        embs['A p_f'] = ((tr_Ap, tr_y), (te_Ap, te_y))

        """
        ### d-svd of p (project onto top d directions of data)
        svd_path = os.path.join(hps.output_dir, f'{hps.model}_pf_trainsvd{prompt_str}.pkl')
        if os.path.exists(svd_path):
            C = pkl.load(open(svd_path, 'rb'))
        else:
            _, _, C = np.linalg.svd(tr_p, full_matrices=False)
            pkl.dump(C, open(svd_path, 'wb'))
        tr_Cp = tr_p.dot(C[:768, :].T)

        svd_path = os.path.join(hps.output_dir, f'{hps.model}_pf_testsvd{prompt_str}.pkl')
        if os.path.exists(svd_path):
            C = pkl.load(open(svd_path, 'rb'))
        else:
            _, _, C = np.linalg.svd(te_p, full_matrices=False)
            pkl.dump(C, open(svd_path, 'wb'))
        te_Cp = te_p.dot(C[:768, :].T)
        embs['dvsd'] = ((tr_Cp, tr_y), (te_Cp, te_y))
        """
    
    return embs

def run_pipeline(emb_mode, embs, hps):
    config_tuple = (hps.task, emb_mode)
    solver, multi = get_solver(config_tuple)
    (tr_x, tr_y), (te_x, te_y) = embs[emb_mode]
    logger.info(f'Fitting on {emb_mode}')
    avg_splits = hps.task == 'mpqa' or hps.task == 'cr'
    if avg_splits:
        logger.info(f'For {hps.task}, we avg results across 10 random train/dev splits instead of using a fixed split')
        all_docs = np.concatenate((tr_x, te_x), 0)
        all_labels = np.concatenate((tr_y, te_y), 0)
        tr_split = int(all_docs.shape[0] * 0.9)
        tr_scores, te_scores = 0., 0.
        for i in range(10):
            inds = np.random.permutation(all_docs.shape[0])
            tr_inds, te_inds = inds[:tr_split], inds[tr_split:]
            tr_x, te_x = all_docs[tr_inds, :], all_docs[te_inds, :]
            tr_y, te_y = all_labels[tr_inds], all_labels[te_inds]
            clf, clf_config = fit_linear_clf(tr_x, tr_y, hps.k_fold, solver=solver, multi_mode=multi)
            tr_score, te_score = clf.score(tr_x, tr_y), clf.score(te_x, te_y)
            tr_scores += tr_score
            te_scores += te_score

        tr_score = tr_scores / 10
        te_score = te_scores / 10
    else:
        clf, clf_config = fit_linear_clf(tr_x, tr_y, hps.k_fold, solver=solver, multi_mode=multi)
        tr_score, te_score = clf.score(tr_x, tr_y), clf.score(te_x, te_y)

    logger.info(f'Train: {tr_score}, Test: {te_score}')    
    return tr_score, te_score, clf_config


def gpt2_batch_embed(docs, model, tokenizer, mode='last', prompt=''):
    tokenizer.pad_token = tokenizer.eos_token
    prompted = [d + prompt for d in docs]
    # list of lengths, to decide the max_len
    enc_dict = tokenizer.batch_encode_plus(prompted, pad_to_max_length=True, max_length=512)
    input_ids = torch.tensor(enc_dict['input_ids'])
    attention_mask = torch.tensor(enc_dict['attention_mask'])

    # Truncate upto maximum length for memory efficiency
    lens = attention_mask.sum(1).numpy()
    max_len = min(lens.max(), 512)
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    with torch.no_grad():
        embs = model.transformer(input_ids=input_ids.to(model.device),
                                 attention_mask=attention_mask.to(model.device))

        embs = embs[0].squeeze()
        embs = embs.cpu().detach().numpy()

    if mode == 'last':
        if len(embs.shape) == 2: # occurs if we have just 1 example in the batch
            embs = np.expand_dims(embs, 0)    
        return embs[np.arange(lens.shape[0]), lens - 1, :]
    if mode == 'mean':
        return embs.sum(axis=1) / lens.numpy()[:, None]
    if mode == 'first':
        return embs[:, 0, :]

def bert_batch_embed(docs, model, tokenizer, mode='first', prompt=''):
    tokenizer.pad_token = tokenizer.eos_token
    prompted = [d + prompt for d in docs]
    # list of lengths, to decide the max_len
    enc_dict = tokenizer.batch_encode_plus(prompted, pad_to_max_length=True, max_length=512)
    input_ids = torch.tensor(enc_dict['input_ids'])
    attention_mask = torch.tensor(enc_dict['attention_mask'])

    # Truncate upto maximum length for memory efficiency
    lens = attention_mask.sum(1).numpy()
    max_len = min(lens.max(), 512)
    input_ids = input_ids[:, :max_len]
    attention_mask = attention_mask[:, :max_len]
    with torch.no_grad():
        embs = model.bert(input_ids=input_ids.to(model.device),
                          attention_mask=attention_mask.to(model.device))
        emb = embs[0]
        emb = model.cls.predictions.transform(emb)
        embs = emb.cpu().detach().numpy()

    if mode == 'last':
        return embs[np.arange(lens.shape[0]), lens - 1, :]
    if mode == 'mean':
        return embs.sum(axis=1) / lens.numpy()[:, None]
    if mode == 'first':
        return embs[:, 0, :]


##### script setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = ArgumentParser()
parser.add_argument('--data_dir', '-d', type=str,
                    default='/n/fs/ptml/datasets/nlp_tasks/')
parser.add_argument('--model', '-m', type=str, default='gpt2',
                    help='Path to model saved using HF. Pass gpt2 for pre-trained GPT-2 model, bert-base-cased for BERT model')
parser.add_argument('--untied', default=False, action='store_true',
                    help='If Phi and the input word embeddings are untied. Defaults to False. Set to false for standard models, true for custom models (e.g. Quad)')
parser.add_argument('--task', '-t', type=str, required=True,
                    help=f'Task to test on')
parser.add_argument('--prompt', '-p', default=False, action='store_true',
                    help='When on, uses the prompt assigned to the task (see code)')
parser.add_argument('--batch_size', '-b', type=int, default=25,
                    help='Batch size for embedding')
parser.add_argument('--output_dir', '-o', type=str, default='',
                    help='Where to save the outputs. If left empty, saves to same directory as data')
parser.add_argument('--run_p', default=False, action='store_true',
                    help='When true, runs clf on full p_f. Requires a lot of memory and time.')
parser.add_argument('--save_p', default=False, action='store_true',
                    help='When true, saves the 50257-dim p_f features for each example')
parser.add_argument('--run_projections', default=False, action='store_true',
                    help='When true, runs clf on random proj and top d directions of data')
# clf arguments
parser.add_argument('--k_fold', '-k', type=int, default=5,
                    help='k value for k-fold CV')

hps = parser.parse_args()

if os.path.exists(hps.model):
    logger.info('The model you have specified is a custom local checkpoint, not a pretrained HF model')
    # corresponds to {folder containing run}_{checkpoint ID}
    if hps.model[-1] == '/':
        hps.model = hps.model[:-1] # strip off trailing slash
    new_path = '_'.join(hps.model.split('/')[-2:])
    logger.info(f'Will refer to this model as {new_path}')
    hps.model_name = new_path
else:
    hps.model_name = hps.model
    
if hps.output_dir == '':
    hps.output_dir = os.path.join(hps.data_dir, hps.task)
assert os.path.exists(os.path.join(hps.data_dir, hps.task)), 'cannot find task in data dir'
if hps.prompt:
    assert hps.task in PROMPTS, f'no prompt defined for {hps.task}'

embs = load_or_generate_embs(hps)
results = {}

results['tr_f'], results['te_f'], results['f_config'] = run_pipeline('f', embs, hps)

## we only need the f results for the BERT model
if 'bert' in hps.model:
    print(results)
    prompt_str = '_prompt' if hps.prompt else ''
    path = os.path.join(hps.output_dir, f'{hps.model}_clf_results{prompt_str}.pkl')
    if os.path.exists(path):
        old_results = pkl.load(open(path, 'rb'))
        old_results.update(results)
        pkl.dump(old_results, open(path, 'wb'))
        print(old_results)
    else:
        pkl.dump(results, open(path, 'wb'))
    import sys;sys.exit()

results['tr_Phipf'], results['te_Phipf'], results['Phipf_config'] = run_pipeline('Phi p_f', embs, hps)

if hps.task in SENTIMENT_TASKS or hps.task == 'agnews':
    results['tr_subpf'], results['te_subpf'], results['subpf_config'] = run_pipeline('subset p_f', embs, hps)
    results['tr_classpf'], results['te_classpf'], results['classpf_config'] = run_pipeline('class p_f', embs, hps)
    if hps.task in SENTIMENT_TASKS:
        results['tr_posnegpf'], results['te_posnegpf'], results['posnegpf_config'] = run_pipeline('posneg p_f', embs, hps)

if hps.run_p:
    results['tr_p'], results['te_p'], results['p_config'] = run_pipeline('p_f', embs, hps)

if hps.run_projections:
    results['tr_Ap'], results['te_Ap'], results['Ap_config'] = run_pipeline('A p_f', embs, hps)
    # we didn't put these in the paper 
    #results['tr_dsvd'], results['te_dsvd'], results['dsvd_config'] = run_pipeline('dsvd', embs, hps)

for k in results.keys():
    if 'te' in k:
        print(f'{k}: {results[k]}')

prompt_str = '_prompt' if hps.prompt else ''
path = os.path.join(hps.output_dir, f'{hps.model_name}_clf_results{prompt_str}.pkl')
if os.path.exists(path):
    old_results = pkl.load(open(path, 'rb'))
    old_results.update(results)
    pkl.dump(old_results, open(path, 'wb'))
    print(old_results)
else:
    pkl.dump(results, open(path, 'wb'))
