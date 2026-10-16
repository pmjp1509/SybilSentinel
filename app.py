#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit interface for the Sybil-attack / defense federated-learning project.

    streamlit run app.py

Tabs: run an experiment with live charts | compare saved runs | saved benchmark figures | how it works.
"""

import glob
import json
import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

from main_fed_avg_sybil import main as run_experiment      # noqa: E402

UI_DIR = os.path.join('results', 'ui')

ATTACKS = {
    'labelflip': 'Label-flip (colluding Sybils train on flipped labels, update boosted)',
    'backdoor': 'Backdoor (trigger patch -> target label 0, update boosted)',
    'signflip': 'Sign-flip (colluding Sybils send the negated, boosted update)',
    'random': 'Random update (one shared random vector)',
    'dpnoise': 'DP-noise attack (original base repo: huge independent noise)',
}
DEFENSES = {
    'none': 'None (plain FedAvg)',
    'filter': 'Similarity filter (OUR defense)',
    'filter_median': 'Filter + median hybrid (OUR defense)',
    'krum': 'Multi-Krum (Blanchard 2017)',
    'median': 'Coordinate-wise median (Yin 2018)',
    'trimmed': 'Trimmed mean (Yin 2018)',
    'foolsgold': 'FoolsGold-style weighting (Fung 2018)',
}
DEF_SHORT = {'none': 'no defense', 'filter': 'similarity filter', 'filter_median': 'filter + median', 'krum': 'Multi-Krum', 'median': 'median',
             'trimmed': 'trimmed mean', 'foolsgold': 'FoolsGold-style'}
COLORS = ['#2e86ab', '#d1495b', '#edae49', '#4a4a4a', '#66a182', '#8d6a9f']

st.set_page_config(page_title='Sybil attacks on FL', page_icon=None, layout='wide')


# ------------------------------------------------------------------ helpers

def stretch(fn, *args, **kwargs):
    """Full-width widget on both new (width='stretch') and old (use_container_width) Streamlit versions."""
    try:
        return fn(*args, width='stretch', **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)

def attack_metric_name(attack, sybil):
    if sybil == 0:
        return 'flip_rate', 'label-flip rate'
    if attack == 'backdoor':
        return 'backdoor_rate', 'backdoor success rate'
    return 'flip_rate', 'label-flip rate'


RUN_COLORS = {'no defense': '#d1495b', 'similarity filter': '#2e86ab', 'filter + median': '#66a182',
              'Multi-Krum': '#edae49', 'median': '#8d6a9f', 'trimmed mean': '#4a4a4a', 'FoolsGold-style': '#b5651d'}


def line_chart(df, y, title, height=300, legend_cols=2):
    if df.empty:
        return None
    runs = list(df['run'].unique())
    colors = [RUN_COLORS.get(r, COLORS[i % len(COLORS)]) for i, r in enumerate(runs)]
    ch = alt.Chart(df).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=25)).encode(
        x=alt.X('round:Q', title='communication round', axis=alt.Axis(tickMinStep=1, format='d')),
        y=alt.Y(y + ':Q', title='%', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('run:N', title=None, scale=alt.Scale(domain=runs, range=colors),
                        legend=alt.Legend(orient='bottom', columns=legend_cols, labelLimit=600)),
        tooltip=['run', 'round', alt.Tooltip(y + ':Q', format='.1f')],
    ).properties(height=height, title=title)
    return ch


def frame(histories, y):
    rows = []
    for label, hist in histories.items():
        for h in hist:
            rows.append({'run': label, 'round': h['round'], y: h[y]})
    return pd.DataFrame(rows)


def build_argv(cfg, defense, out_dir):
    argv = ['--epochs', str(cfg['rounds']), '--sybil', str(cfg['sybil']), '--attack', cfg['attack'],
            '--defense', defense, '--iid', '1' if cfg['iid'] else '0', '--seed', str(cfg['seed']),
            '--sybil_boost', str(cfg['boost']), '--sybil_jitter', str(cfg['jitter']),
            '--filter_threshold', str(cfg['threshold']), '--epsilon', str(cfg['epsilon']),
            '--dp', '1' if cfg['dp'] else '0', '--out_dir', out_dir, '--local_ep', str(cfg['local_ep']),
            '--dataset', 'mnist', '--model', 'cnn']
    if cfg['stealth_dp']:
        argv.append('--sybil_dp')
    return argv


def load_saved_runs():
    runs = {}
    for pat in (os.path.join(UI_DIR, '*.json'), os.path.join('results', '*.json'),
                os.path.join('results', 'compare', '*.json')):
        for f in sorted(glob.glob(pat)):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            runs[os.path.relpath(f, 'results')] = r
    return runs


def short_label(r):
    c = r['config']
    d = c.get('defense', 'filter' if c.get('enable_filter') else 'none')
    atk = 'clean' if c['sybil'] == 0 else '%d%% %s' % (c['sybil'] * 100 // c['num_users'], c['attack'])
    if c['sybil'] and c.get('sybil_jitter', 0.2) != 0.2:
        atk += ' (stealth %.1f)' % c['sybil_jitter']
    return '%s, %s, %dr' % (DEF_SHORT.get(d, d), atk, c['epochs'])


def describe(r):
    c = r['config']
    d = c.get('defense', 'filter' if c.get('enable_filter') else 'none')
    pct = c['sybil'] * 100 // c['num_users']
    atk = 'no attack' if c['sybil'] == 0 else '%s%s' % (c['attack'], ' (jitter %.1f)' % c['sybil_jitter']
                                                           if c.get('sybil_jitter', 0.2) != 0.2 else '')
    return '%s | %d%% Sybils (%s) | %d rounds' % (DEF_SHORT.get(d, d), pct, atk, c['epochs'])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header('Experiment settings')
    st.caption('MNIST, 100 clients, 10 sampled per round, CNN model, Laplace DP noise on honest clients.')
    rounds = st.slider('Communication rounds', 3, 100, 15, help='More rounds = better accuracy but slower.')
    sybil = st.slider('Sybil clients (out of 100)', 0, 60, 30,
                      help='Fake clients controlled by ONE adversary. 30 = 30% of the population.')
    attack = st.selectbox('Attack', list(ATTACKS), format_func=ATTACKS.get, disabled=(sybil == 0))
    defense = st.selectbox('Defense', list(DEFENSES), index=1, format_func=DEFENSES.get)
    compare = st.checkbox('Also run the no-defense baseline', value=True,
                          help='Runs a second experiment with identical settings and no defense, for comparison.',
                          disabled=(defense == 'none'))
    with st.expander('Advanced attack settings'):
        boost = st.slider('Update boost (attack strength)', 1.0, 6.0, 3.0, 0.5)
        jitter = st.slider('Sybil stealth: independent noise between Sybils', 0.0, 2.0, 0.2, 0.1,
                           help='0.2 = Sybils are ~96% similar (easy to spot). 1.0 = ~50% similar: evades the '
                                'similarity filter. This is the adaptive-attacker test.')
        stealth_dp = st.checkbox('Sybils also add honest-level DP noise', value=False)
    with st.expander('Advanced defense / training settings'):
        threshold = st.slider('Filter threshold tau', 0.5, 0.99, 0.90, 0.01,
                              help='Only used by the similarity filter: updates more similar than this to another '
                                   'client are dropped.')
        dp = st.checkbox('Honest clients add DP noise', value=True)
        epsilon = st.slider('DP epsilon (smaller = more noise)', 1.0, 20.0, 8.0, 1.0)
        local_ep = st.slider('Local epochs per client', 1, 5, 2)
        iid = st.checkbox('IID data split', value=True)
        seed = st.number_input('Random seed', 0, 9999, 1)
    run_clicked = stretch(st.button, 'Run experiment', type='primary')

cfg = dict(rounds=rounds, sybil=sybil, attack=attack, boost=boost, jitter=jitter, stealth_dp=stealth_dp,
           threshold=threshold, dp=dp, epsilon=epsilon, local_ep=local_ep, iid=iid,
           seed=int(seed))

# ------------------------------------------------------------------ page
st.title('Sybil attacks on federated learning')
st.markdown('One adversary creates many fake clients to corrupt the shared model. Choose an attack and a defense on the '
            'left, then run it and watch accuracy and attack success live.')

tab_run, tab_cmp, tab_bench, tab_how = st.tabs(['Run experiment', 'Compare runs', 'Saved benchmark', 'How it works'])

# ------------------------------------------------------------------ tab: run
with tab_run:
    if 'last' not in st.session_state:
        st.session_state.last = None

    if run_clicked:
        todo = [defense] + (['none'] if (compare and defense != 'none') else [])
        histories = {}
        yname, ylabel = attack_metric_name(attack, sybil)
        prog = st.progress(0.0, text='Starting...')
        status = st.empty()
        c1, c2 = st.columns(2)
        ph_acc, ph_atk = c1.empty(), c2.empty()
        results = {}
        for k, d in enumerate(todo):
            label = DEF_SHORT[d]
            histories[label] = []

            def cb(info, label=label, k=k, d=d):
                histories[label].append(info)
                frac = (k + (info['round'] + 1) / cfg['rounds']) / len(todo)
                prog.progress(min(frac, 1.0), text='%s: round %d / %d' % (label, info['round'] + 1, cfg['rounds']))
                bits = 'Round %d: %d Sybil(s) among 10 sampled clients' % (info['round'] + 1, info['n_sybil'])
                if d in ('filter', 'filter_median'):
                    bits += ' | filter dropped %d Sybil(s), %d honest' % (info['flagged_sybil'], info['flagged_honest'])
                status.caption('%s: %s | accuracy %.1f%%' % (label, bits, info['acc']))
                stretch(ph_acc.altair_chart, line_chart(frame(histories, 'acc'), 'acc', 'Test accuracy (%)'))
                stretch(ph_atk.altair_chart, line_chart(frame(histories, yname), yname, ylabel + ' (%, lower is better)'))

            results[d] = run_experiment(build_argv(cfg, d, UI_DIR), progress_cb=cb)
        prog.progress(1.0, text='Done')
        status.empty()
        st.session_state.last = dict(results=results, yname=yname, ylabel=ylabel, histories=histories)

    last = st.session_state.last
    if last is None:
        st.info('Set the options in the sidebar and press **Run experiment**. A 15-round run takes roughly a minute or '
                'two on a laptop CPU.')
    else:
        if not run_clicked:                     # re-draw the final charts after a re-run of the script
            c1, c2 = st.columns(2)
            stretch(c1.altair_chart, line_chart(frame(last['histories'], 'acc'), 'acc', 'Test accuracy (%)'))
            stretch(c2.altair_chart, line_chart(frame(last['histories'], last['yname']), last['yname'],
                                       last['ylabel'] + ' (%, lower is better)'))
        st.subheader('Summary')
        cols = st.columns(len(last['results']))
        for col, (d, r) in zip(cols, last['results'].items()):
            s = r['summary']
            atk_val = s['mean_last5_backdoor'] if last['yname'] == 'backdoor_rate' else s['final_flip_rate']
            with col:
                st.markdown('**%s**' % DEF_SHORT[d])
                st.metric('Accuracy (last 5 rounds)', '%.1f%%' % s['mean_last5_acc'])
                st.metric(last['ylabel'], '%.1f%%' % atk_val)
                if d in ('filter', 'filter_median') and s['tpr'] is not None:
                    st.metric('Sybils caught / honest wrongly dropped', '%.0f%% / %.1f%%' % (100 * s['tpr'], 100 * s['fpr']))
        st.caption('Runs are saved in results/ui/ and can be compared in the next tab.')

# ------------------------------------------------------------------ tab: compare
with tab_cmp:
    saved = load_saved_runs()
    if not saved:
        st.info('No saved runs yet. Run an experiment first (results/ui/) or run benchmark.py / benchmark_compare.py.')
    else:
        keys = list(saved)
        default = [k for k in keys if k.startswith('ui')][-3:] or keys[:3]
        pick = st.multiselect('Runs to compare', keys, default=default,
                              format_func=lambda k: '%s  [%s]' % (describe(saved[k]), k))
        if pick:
            hist, seen = {}, {}
            for k in pick:
                lab = short_label(saved[k])
                seen[lab] = seen.get(lab, 0) + 1
                if seen[lab] > 1:
                    lab += ' #%d' % seen[lab]
                n = len(saved[k]['acc'])
                hist[lab] = [{'round': i, 'acc': saved[k]['acc'][i], 'flip_rate': saved[k]['flip_rate'][i],
                              'backdoor_rate': saved[k].get('backdoor_rate', [0.0] * n)[i]} for i in range(n)]
            a, b, c = st.columns(3)
            stretch(a.altair_chart, line_chart(frame(hist, 'acc'), 'acc', 'Test accuracy (%)', legend_cols=1))
            stretch(b.altair_chart, line_chart(frame(hist, 'flip_rate'), 'flip_rate', 'Label-flip rate (%)', legend_cols=1))
            stretch(c.altair_chart, line_chart(frame(hist, 'backdoor_rate'), 'backdoor_rate', 'Backdoor success rate (%)', legend_cols=1))
            rows = []
            for k in pick:
                s = saved[k]['summary']
                rows.append({'run': describe(saved[k]), 'file': k, 'accuracy (last 5)': round(s['mean_last5_acc'], 2),
                             'final label-flip %': round(s['final_flip_rate'], 2),
                             'final backdoor %': round(s.get('final_backdoor_rate', 0.0), 2),
                             'TPR': None if s['tpr'] is None else round(s['tpr'], 2),
                             'FPR': None if s['fpr'] is None else round(s['fpr'], 3)})
            stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ tab: saved benchmark
with tab_bench:
    shown = False
    for title, folder, imgs, md in (
            ('Attack scale benchmark (0 / 20 / 40% Sybils, with and without our filter)', 'results',
             ['acc_vs_rounds.png', 'acc_vs_sybil.png', 'detection.png'], 'summary.md'),
            ('Defense comparison (our filter vs Krum, median, trimmed mean, FoolsGold-style)',
             os.path.join('results', 'compare'), ['compare_accuracy.png', 'compare_attack_success.png'],
             'compare_summary.md')):
        paths = [os.path.join(folder, i) for i in imgs if os.path.exists(os.path.join(folder, i))]
        if not paths:
            continue
        shown = True
        st.subheader(title)
        if len(paths) == 3:                       # wide per-round figure, then the two small bar charts side by side
            stretch(st.image, paths[0])
            cc = st.columns(2)
            stretch(cc[0].image, paths[1])
            stretch(cc[1].image, paths[2])
        else:
            for p in paths:
                stretch(st.image, p)
        mdp = os.path.join(folder, md)
        if os.path.exists(mdp):
            st.markdown(open(mdp).read())
    if not shown:
        st.info('Run `python benchmark.py` and `python benchmark_compare.py` to generate the saved figures.')

# ------------------------------------------------------------------ tab: how it works
with tab_how:
    st.markdown("""
### The setting
Federated learning: many clients train a shared model on their own data and send only model updates. The server
averages them (**FedAvg**). Honest clients add Laplace noise for differential privacy (from the base repo
*ITSEG-MQ/Sybil-Attacks-and-Defense-on-DP-FL*).

### The attack
One adversary controls many fake clients (**Sybils**). They send near-copies of one malicious update, so they carry
many "votes" in the average.
- **Label-flip**: train on flipped labels, boost the update. Success is measured by the *label-flip rate*.
- **Backdoor**: teach the model that a small white square means "digit 0" while keeping normal accuracy high.
  Success is measured by the *backdoor success rate*.
- **Stealth** (advanced): Sybils add independent noise to look different from each other (adaptive attacker).

### Our defense: similarity filter
Honest updates differ (different data, different DP noise); colluding Sybils point the same way.
1. update = client weights minus global weights, flattened to a vector
2. cosine similarity between all pairs of clients
3. score = each client's maximum similarity to any other client
4. score above the threshold (default 0.9) means suspected Sybil, so drop it
5. FedAvg on the rest

### Comparison defenses
**Multi-Krum** keeps the updates closest to their neighbours. **Median** and **trimmed mean** take robust
per-coordinate statistics. **FoolsGold-style** down-weights clients that are too similar (stateless per-round variant).

### Limits to remember
A single Sybil in a round has no similar partner, so the filter cannot see it. Sybils that add enough independent
noise can slip under a fixed threshold. Results come from one seed on MNIST.
""")
