#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import yaml
import subprocess
import logging

import pandas as pd

from datetime import datetime

from scipy.stats import ttest_rel
from sklearn.metrics import f1_score


context_masks = {
    'TARG': {0},
    'L$_1$': {-1},
    'L$_2$': {-2, -1},
    'L$_3$': {-3, -2, -1},
    'R$_1$': {1},
    'R$_2$': {2, 1},
    'R$_3$': {3, 2, 1},
    'B$_1$': {-1, 1},
    'B$_2$': {-2, -1, 1, 2},
    'B$_3$': {-3, -2, -1, 1, 2, 3},
    'ALL_MASK': set(range(-100, 101)),
    'B_ALL': set(range(-100, 101)) - {0},
    'L_ALL': set(range(-100, 0)),
    'R_ALL': set(range(1, 101)),
}


def quick_load_experiments_tsv(exp_dir):
    exp_tsv = os.path.join(exp_dir, "experiments.tsv")
    if os.path.exists(exp_tsv):
        logging.info(f"Loading experiments.tsv from {exp_dir}")
        df = pd.read_table(exp_tsv, sep="\t")
        for col in df.columns:
            if 'running_time' in col:
                df[col] = pd.to_timedelta(df[col])
            elif '_time' in col:
                df[col] = pd.to_datetime(df[col])
        return df
    else:
        logging.warning(f"File {exp_tsv} not found")


def get_recently_modified(exp_dir, date):
    td = (datetime.utcnow() - date).total_seconds() / 60
    td = int(td) - 1
    cmd = f"find {exp_dir} -mmin -{td}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    out, _ = p.communicate()
    return out.decode('utf8').strip()


def load_experiment_dirs(exp_dir, compute_F_score=True, always_reload=False):
    exp_tsv = os.path.join(exp_dir, "experiments.tsv")
    if always_reload is False:
        if os.path.exists(exp_tsv):
            edate = pd.to_datetime(os.path.getmtime(exp_tsv), unit='s')
            mod = get_recently_modified(exp_dir, edate)
            if not mod:
                logging.info(f"Loading experiments.tsv from {exp_dir}")
                df = pd.read_table(exp_tsv, sep="\t")
                for col in df.columns:
                    if 'running_time' in col:
                        df[col] = pd.to_timedelta(df[col])
                    elif '_time' in col:
                        df[col] = pd.to_datetime(df[col])
                return df

    logging.info(f"Reading experiments from dir {exp_dir}")
    exps = []
    file_cache = {}
    for fn in sorted(os.scandir(exp_dir), key=lambda s: s.path):
        if not os.path.exists(os.path.join(fn.path, "result.yaml")):
            continue
        with open(os.path.join(fn.path, "config.yaml")) as f:
            exp_d = yaml.load(f, Loader=yaml.Loader)
        with open(os.path.join(fn.path, "result.yaml")) as f:
            exp_d.update(yaml.load(f, Loader=yaml.Loader))

        for split in ['train', 'dev', 'test']:
            if compute_F_score:
                if split == 'test':
                    gold_fn = exp_d['train_file'].replace('train', 'test')
                else:
                    gold_fn = exp_d[f'{split}_file']
                out_fn = os.path.join(fn.path, f"{split}.out")
                if os.path.exists(out_fn):
                    if gold_fn not in file_cache:
                        colnum = pd.read_table(
                            gold_fn, quoting=3, na_filter=False).shape[1]
                        if colnum > 5:
                            names = list(range(colnum))
                            names[-3] = 'label'
                        else:
                            names = list(range(colnum-1)) + ['label']
                    gold = file_cache.setdefault(
                        gold_fn, pd.read_table(gold_fn, names=names,
                                               quoting=3, na_filter=False))
                    pred = pd.read_table(out_fn, names=names, quoting=3,
                                         na_filter=False)
                    if len(pred) != len(gold):
                        logging.warning(f"{out_fn}: prediction size differs "
                                        "from gold size")
                    else:
                        exp_d[f'{split}_F_score'] = f1_score(
                            gold['label'], pred['label'], average='macro')
            acc_fn = os.path.join(fn.path, f'{split}.word_accuracy')
            if f'{split}_acc' in exp_d:
                exp_d[f'{split}_acc_list'] = exp_d[f'{split}_acc']
            if os.path.exists(acc_fn):
                with open(acc_fn) as f:
                    try:
                        exp_d[f'{split}_acc'] = float(f.read())
                    except ValueError:
                        logging.warning("Unable to read accuracy file: "
                                        f"{os.path.abspath(acc_fn)}")

        exp_d['experiment_dir'] = os.path.realpath(fn.path)
        exps.append(exp_d)

    exps = pd.DataFrame(exps)
    if 'running_time' in exps.columns:
        exps['running_time'] = pd.to_timedelta(exps['running_time'], unit='s')
    exps.to_csv(exp_tsv, sep="\t", index=False)
    for col in exps.columns:
        if 'running_time' in col:
            exps[col] = pd.to_timedelta(exps[col])
        elif '_time' in col:
            exps[col] = pd.to_datetime(exps[col])
    return exps


def get_significance_by_column(data, p=0.05):
    sign = []
    for c1 in data.columns:
        for c2 in data.columns:
            if c2 == c1:
                continue
            d = data[[c1, c2]].dropna()
            s = {
                "col1": c1,
                "col2": c2,
                "dof": len(d[c1])-1,
                f"mean1": data[c1].mean(),
                f"mean2": data[c2].mean(),
                }
            r = ttest_rel(d[c1], d[c2])
            s['p'] = r[1]
            sign.append(s)
    sign = pd.DataFrame(sign)
    sign['sign'] = sign.p < p
    return sign


def does_subword_choice_matter(perturbation):
    if perturbation in ('permute', 'mBERT-emb', 'mBERT-char', 'mBERT-rand-char',
                        'mBERT-rand-subw', 'mBERT-rand-subw-emb', 'mBERT-nocontext'):
        return True
    if 'prev' in perturbation or 'next' in perturbation:
        return True
    return False


def is_model_with_subword_choice(model):
    return model in ('mBERT', 'chLSTM', 'swLSTM')


def is_probing_location_better(row, better_location):
    if row.model == 'chLSTM-rand':
        return better_location.loc[('chLSTM', row.language, row.task)] == row.probing_location
    # If subword choice doesn't matter for a model, either way is a good choice
    if not is_model_with_subword_choice(row.model):
        return True
    if row.model == 'mBERT':
        if not pd.isna(row.perturbation):
            if 'next' in row.perturbation:
                return row.probing_location == 'first'
            if 'prev' in row.perturbation:
                return row.probing_location == 'last'
        # We use a single [MASK] symbol in place of the masked target
        # so subword choice is not really a choice.
        if 'mask_positions' in row and 0 in row.mask_positions:
            return True
    return better_location.loc[(row.model, row.language, row.task)] == row.probing_location


def add_probing_location_better_column(exps, col_name='probing_location_better'):
    models_with_subword_choice = set(m for m in exps.model.unique()
                                     if is_model_with_subword_choice(m))
    no_pert = exps[
        (exps.model.isin(models_with_subword_choice)) &
        (exps.perturbation.isnull()) &
        (exps.probing_location.isin(('first', 'last')))
    ]
    g = no_pert.groupby(['model', 'language', 'task', 'probing_location'])
    dev_acc = g['dev_acc'].mean().unstack()
    # When the first and the last subword are equal, we pick the last
    dev_acc['better'] = 'last'
    dev_acc.loc[(dev_acc['first']>dev_acc['last']), 'better'] = 'first'
    dev_better = dev_acc['better']
    exps[col_name] = exps.apply(is_probing_location_better, axis=1, better_location=dev_better)
    return exps


def all_conditions_true(row, condition_dict):
    for cond, value in condition_dict.items():
        if row[cond] != value and str(row[cond]) != str(value):
            return False
    return True


def add_model_and_perturbation_fields(row):
    models = []
    perturbations = []
    if row.model == 'EmbeddingClassifier' and row.dataset_class == 'EmbeddingProberDataset':
        models.append('fastText')
        perturbations.append(None)
    # old WLSTM
    if row.model == 'SequenceClassifier' and row.dataset_class == 'WordOnlySentenceProberDataset':
        models.append('WLSTM-old')
        perturbations.append(None)
    # old SLSTM
    if row.model == 'MidSequenceClassifier' and row.dataset_class == 'MidSentenceProberDataset':
        if row.freeze_lstm_encoder:
            models.append('chLSTM-rand')
        else:
            models.append('chLSTM')
        perturbations.append(None)
    # new SLSTM. Subword or character tokenization
    if row.dataset_class == 'SLSTMDataset':
        if row.external_tokenizer == 'bert-base-multilingual-cased':
            models.append('swLSTM')
            perturbations.append(None)
        else:
            if row.freeze_lstm_encoder:
                models.append('chLSTM-rand')
            else:
                models.append('chLSTM')
            perturbations.append(None)
    # new WLSTM. Subword or character tokenization
    if row.dataset_class == 'WLSTMDataset':
        if row.external_tokenizer == 'bert-base-multilingual-cased':
            models.append('swWLSTM')
            perturbations.append(None)
        else:
            models.append('chWLSTM')
            perturbations.append(None)
    # we should have covered all non-BERT baselines
    # assert row.model_name == 'bert-base-multilingual-cased'
    # assert row.model == 'SentenceRepresentationProber'
    if not models:
        models.append('mBERT')

    # finetuned models are not considered a perturbation
    if row.train_base_model:
        perturbations.append('mBERT-finetuned')

    # assert row.train_base_model is False

    unperturbed_config = {
        'shift_target': 0,
        'mask_positions': set(),
        'bow': False,
        'use_character_tokenization': False,
        'layer_pooling': 'weighted_sum',
        'randomize_embedding_weights': False,
        'target_only': False,
        'train_base_model': False,
    }
    row.mask_positions = set(eval(str(row.mask_positions)))

    # unperturbed
    if all_conditions_true(row, unperturbed_config):
        perturbations.append(None)

    # PREV
    for c in range(1, 5):
        cfg = unperturbed_config.copy()
        cfg['shift_target'] = -c
        cfg['subword_pooling'] = 'last'
        if all_conditions_true(row, cfg):
            perturbations.append(f'prev{c}')

        cfg = unperturbed_config.copy()
        cfg['shift_target'] = -c
        cfg['subword_pooling'] = 'first'
        if all_conditions_true(row, cfg):
            perturbations.append(f'mBERT-invalid-shift')

        cfg = unperturbed_config.copy()
        cfg['shift_target'] = c
        cfg['subword_pooling'] = 'first'
        if all_conditions_true(row, cfg):
            perturbations.append(f'next{c}')

        cfg = unperturbed_config.copy()
        cfg['shift_target'] = c
        cfg['subword_pooling'] = 'last'
        if all_conditions_true(row, cfg):
            perturbations.append(f'mBERT-invalid-shift')

    cfg = unperturbed_config.copy()
    cfg['layer_pooling'] = '0'
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-emb')

    cfg = unperturbed_config.copy()
    cfg['layer_pooling'] = '0'
    cfg['randomize_embedding_weights'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-rand-subw-emb')

    cfg = unperturbed_config.copy()
    cfg['randomize_embedding_weights'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-rand-subw')

    cfg = unperturbed_config.copy()
    cfg['randomize_embedding_weights'] = True
    cfg['target_only'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-rand-target-only')

    cfg = unperturbed_config.copy()
    cfg['use_character_tokenization'] = 'full'
    cfg['randomize_embedding_weights'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-rand-char')

    cfg = unperturbed_config.copy()
    cfg['use_character_tokenization'] = 'target_only'
    cfg['randomize_embedding_weights'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-rand-chartarget')

    cfg = unperturbed_config.copy()
    cfg['use_character_tokenization'] = 'full'
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-char')

    cfg = unperturbed_config.copy()
    cfg['use_character_tokenization'] = 'full'
    cfg['target_only'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-char-discard-context')

    cfg = unperturbed_config.copy()
    cfg['use_character_tokenization'] = 'target_only'
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-chartarget')

    cfg = unperturbed_config.copy()
    cfg['target_only'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-nocontext')

    cfg = unperturbed_config.copy()
    cfg['target_only'] = True
    cfg['layer_pooling'] = 0
    if all_conditions_true(row, cfg):
        perturbations.append('mBERT-nocontext-emb')

    cfg = unperturbed_config.copy()
    cfg['bow'] = True
    if all_conditions_true(row, cfg):
        perturbations.append('permute')

    for name, mask in context_masks.items():
        cfg = unperturbed_config.copy()
        cfg['mask_positions'] = mask
        if all_conditions_true(row, cfg):
            perturbations.append(name)

    cfg = unperturbed_config.copy()
    del cfg['mask_positions']
    if all_conditions_true(row, cfg) and len(row.mask_positions) > 0:
        if not set(context_masks.keys()) & set(perturbations):
            perturbations.append('other-masking')

    if not perturbations:
        perturbations.append('UNKNOWN')
    row['models'] = models
    row['perturbations'] = perturbations
    return row
