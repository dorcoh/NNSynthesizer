import sys

import pandas as pd

def create_props(df):
    props = df['exp_key'].str.split('::', expand=True)
    props.rename(columns=lambda x: props.iloc[0][x].split('-')[0], inplace=True)
    props = props.apply(lambda x: x.str.split('-').str[1], axis=0)
    return props

path = sys.argv[1]
output_fname = sys.argv[2]
df = pd.read_csv(path)

props = create_props(df)
a = df.merge(props, left_index=True, right_index=True)
ext = a['extra_params']
a.drop(['RepairResult', 'exp_key', 'extra_params'], axis=1, inplace=True)
a = a.assign(**{'extra_params': ext})

# fill unsat if time!='timeout'

if False:
    def fill_unsat(x, col):
        import math
        if not str(x['time']) == 'timeout' and not isinstance(x['avg_acc_before'], str) and not isinstance(x['avg_acc_after'], str) and math.isnan(x['avg_acc_before']) and math.isnan(x['avg_acc_after']):
            return 'unsat'
        else:
            return x[col]

    after = a.apply(lambda x: fill_unsat(x, 'avg_acc_after'), axis=1)
    before = a.apply(lambda x: fill_unsat(x, 'avg_acc_before'), axis=1)

    a['avg_acc_before'] = before
    a['avg_acc_after'] = after

# remove configurations which were deprecated (all NaN and only config_path is filled)
if True:
    print(a.shape)
    def deprecate(row, columns):
        cols = [x for x in columns if x not in ['config_path', 'exp_id']]
        for col in cols:
            if not pd.isna(row[col]):
                # at least one col is not nan
                return 0
        # all checked columns are nan
        return 1
    a['deprecate'] = a.apply(lambda row: deprecate(row, a.columns), axis=1)
    mask = a['deprecate'] == 0
    a = a[mask]
    cols = [x for x in a.columns if x != 'deprecate']
    a = a[cols]
    a.reset_index(inplace=True, drop=True)

print(a.shape)
a[cols].to_csv(output_fname, index=False)