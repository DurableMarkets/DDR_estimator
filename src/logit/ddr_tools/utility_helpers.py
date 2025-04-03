
def construct_utility_colnames(utility_type, variable_str, specification, options):
    if specification[utility_type] is not None:
        nconsumers, ncartypes = specification[utility_type]
    else:
        nconsumers = 0
        ncartypes = 0

    if (nconsumers == 0) | (ncartypes == 0):
        cols = []
        cols_for_utility_funcs = []
    elif (nconsumers == 1) & (ncartypes == 1):
        cols = [variable_str.format("all", "all")]
        cols_for_utility_funcs=[
            [
                variable_str.format(all, 'all')
                for _ in range(1, options["n_car_types"] + 1)
            ]
            for _ in range(0, options["n_consumer_types"])
        ]
    elif (nconsumers == 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, "all")
                for ncartype in range(1, options["n_car_types"] + 1)
            ]
        ]
        cols_for_utility_funcs=[
            [
                variable_str.format(ncartype, 'all')
                for ncartype in range(1, options["n_car_types"] + 1)
            ]
            for _ in range(0, options["n_consumer_types"])
        ]
    elif (nconsumers > 1) & (ncartypes == 1):
        cols = [
            [variable_str.format("all", nconsumer)]
            for nconsumer in range(0, options["n_consumer_types"])
        ]
        cols_for_utility_funcs = [
            [
                variable_str.format('all', nconsumer)
                for _ in range(1, options["n_car_types"] + 1)
            ]
            for nconsumer in range(0, options["n_consumer_types"])
        ]
    elif (nconsumers > 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, nconsumer)
                for ncartype in range(1, options["n_car_types"] + 1)
            ]
            for nconsumer in range(0, options["n_consumer_types"])
        ]
        cols_for_utility_funcs = cols
    else:
        raise ValueError(
            "Invalid specification chosen for utility type {}".format(utility_type)
        )

    if (nconsumers == 1) & (ncartypes == 1):
        cols_flat = cols
    else:
        cols_flat = [item for sublist in cols for item in sublist]
    
    return cols, cols_flat, cols_for_utility_funcs


