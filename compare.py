"""Compare experiment runs side by side: python compare.py base onestep beam cubie symaug [--last]

Reads cache/<tag>/net_metrics.json for every tag and prints, per run, the final (or best) values of
the metrics that matter for the memorise-or-learn question, plus the highest curriculum depth reached.
"""
import argparse
import json
import os

import numpy as np

COLS = [('K', 'K', lambda r: '%d' % r['K']),
        ('greedy', '贪心解出', lambda r: '%.1f%%' % (100 * r['success_random'])),
        ('beam', '束搜索解出', lambda r: '%.1f%%' % (100 * r['beam_random']) if r.get('beam_random') is not None else '-'),
        ('onestep', '单步准确', lambda r: '%.1f%%' % (100 * r['onestep_random']) if r.get('onestep_random') is not None else '-'),
        ('unseen', '未见过解出', lambda r: '%.1f%%' % (100 * r['unseen_success']) if r.get('unseen_success') is not None else '-'),
        ('sampled', '已采样', lambda r: '%.1f%%' % (100 * r['coverage'])),
        ('|Q+d|', '价值误差', lambda r: '%.3f' % r['q_error']),
        ('d7', '深度7贪心', lambda r: '%.0f%%' % (100 * r['success_by_depth'][6])),
        ('d9', '深度9贪心', lambda r: '%.0f%%' % (100 * r['success_by_depth'][8])),
        ('updates', '更新', lambda r: format(r.get('updates', 0), ',')),
        ('time', '时间', lambda r: '%d:%02d' % (r['time'] // 60, r['time'] % 60))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--best', action='store_true', help='best evaluation of each run instead of the last')
    ap.add_argument('--cache', default='cache')
    args = ap.parse_args()
    rows = []
    for tag in args.tags:
        path = os.path.join(args.cache, tag, 'net_metrics.json')
        if not os.path.exists(path):
            rows.append((tag, None, None, None)); continue
        d = json.load(open(path))
        recs = d['records']
        r = max(recs, key=lambda x: x['success_random']) if args.best else recs[-1]
        rows.append((tag, r, max(x['K'] for x in recs), d['config']))
    head = '%-14s' % 'run' + ''.join('%12s' % c[0] for c in COLS) + '%8s' % 'maxK'
    print(head)
    print('-' * len(head))
    for tag, r, maxk, cfg in rows:
        if r is None:
            print('%-14s  (no cache/%s/net_metrics.json)' % (tag, tag)); continue
        print('%-14s' % tag + ''.join('%12s' % c[2](r) for c in COLS) + '%8d' % maxk)
    print()
    for tag, r, maxk, cfg in rows:
        if cfg:
            keys = ['features', 'promote_by', 'symmetry_aug', 'layers', 'batch', 'lr', 'lr_final', 'promote', 'k_margin', 'updates', 'minutes', 'seed']
            print('  %-12s ' % tag + '  '.join('%s=%s' % (k, cfg.get(k)) for k in keys if k in cfg))


if __name__ == '__main__':
    main()
