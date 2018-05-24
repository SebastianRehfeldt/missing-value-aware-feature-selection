import gurobipy as gb


def deduce_relevances(features, knowledgebase):
    gb.setParam('OutputFlag', 0)
    m = gb.Model('rar')

    solver_variables = {}
    for feature in features:
        solver_variables[feature] = m.addVar(
            name=feature, vtype=gb.GRB.CONTINUOUS, lb=0)

    vars_average = m.addVar(name='s', vtype=gb.GRB.CONTINUOUS)
    vars_sum = sum(solver_variables.values())

    m.setObjective(
        vars_sum + _squared_dist(solver_variables.values(), vars_average),
        gb.GRB.MINIMIZE)

    m.addConstr(vars_average == (vars_sum / len(features)))

    for subset in knowledgebase:
        objective_vars = [
            solver_variables[feature] for feature in subset["features"]
        ]
        objective_sum = sum(objective_vars)
        m.addConstr(objective_sum >= subset["score"]["relevance"])

        # TODO: discuss extra constraint/ value for single features?
        """
        if len(objective_vars) == 1:
            m.addConstr(objective_sum <= 3 * subset["score"]["relevance"])
        """

    m.optimize()

    single_relevances = {}
    for v in m.getVars():
        if v.varName in features:
            single_relevances[v.varName] = v.x
    return single_relevances


def _squared_dist(variables, mean):
    return sum(map(lambda v: (v - mean) * (v - mean), variables))
