# %%
import numpy as np
import pandas as pd
from project.utils.data_loader import DataLoader
from project.utils.data_modifier import introduce_missing_values
from project.utils.data_scaler import scale_data

data_loader = DataLoader()
data = data_loader.load_data("iris", "arff")
data = introduce_missing_values(data)
data = scale_data(data)
data.X.head()

test_X = data.X.iloc[0:10, :]
test_y = data.y.iloc[0:10]
test = data.replace(X=test_X, y=test_y, shape=test_X.shape)

train_X = data.X.iloc[10:, :].reset_index(drop=True)
train_y = data.y.iloc[10:].reset_index(drop=True)
train = data.replace(X=train_X, y=train_y, shape=train_X.shape)

# %%


def data2table(data):
    import numpy as np
    from Orange.data.variable import DiscreteVariable, ContinuousVariable
    from Orange.data import Domain, Table
    f_array = []
    for i, t in enumerate(data.f_types):
        f_name = data.X.columns[i]
        if t == "numeric":
            f_array.append(ContinuousVariable(f_name))
        else:
            values = [v.decode('utf-8') for v in np.unique(data.X[f_name])]
            f_array.append(DiscreteVariable(f_name, values=values))

    l_name = data.y.name
    if data.l_type == "nominal":
        values = [v.decode('utf-8') for v in np.unique(data.y)]
        class_var = DiscreteVariable(f_name, values=values)
    else:
        class_var = ContinuousVariable(l_name)

    types = data.f_types.append(pd.Series(data.l_type, [data.y.name]))
    nominal_features = types.loc[types == "nominal"].index.tolist()
    combined = data.X
    combined[data.y.name] = data.y.values

    str_df = combined[nominal_features]
    str_df = str_df.stack().str.decode('utf-8').unstack()

    for col in str_df:
        combined[col] = str_df[col]

    domain = Domain(f_array, class_vars=class_var)
    table = Table.from_list(domain=domain, rows=combined.values.tolist())
    return table


train_table = data2table(train)
test_table = data2table(test)


# %%
from Orange.classification import TreeLearner
classifier = TreeLearner().fit_storage(train_table)
print(test_table.Y)
classifier(test_table.X)


# %%
from Orange.preprocess.score import ReliefF
scores = ReliefF(train_table).score_data
for attr, score in zip(train_table.domain.attributes, scores):
    print('%.3f' % score, attr.name)


# %%
from Orange.distance import Euclidean
dist_model = Euclidean(normalize=True).fit(train_table)
print(train_table.X.shape)
dist_model(train_table)
