
def construct_utility_colnames(utility_type, variable_str, specification, options):
    if specification[utility_type] is not None:
        nconsumers, ncartypes = specification[utility_type]
    else:
        nconsumers = 0
        ncartypes = 0

    if (nconsumers == 0) | (ncartypes == 0):
        cols = []
    elif (nconsumers == 1) & (ncartypes == 1):
        cols = [variable_str.format("all", "all")]
    elif (nconsumers == 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, "all")
                for ncartype in range(1, options["num_car_types"] + 1)
            ]
        ]
    elif (nconsumers > 1) & (ncartypes == 1):
        cols = [
            [variable_str.format("all", nconsumer)]
            for nconsumer in range(0, options["num_consumer_types"])
        ]
    elif (nconsumers > 1) & (ncartypes > 1):
        cols = [
            [
                variable_str.format(ncartype, nconsumer)
                for ncartype in range(1, options["num_car_types"] + 1)
            ]
            for nconsumer in range(0, options["num_consumer_types"])
        ]
    else:
        raise ValueError(
            "Invalid specification chosen for utility type {}".format(utility_type)
        )

    if (nconsumers == 1) & (ncartypes == 1):
        cols_flat = cols
    else:
        cols_flat = [item for sublist in cols for item in sublist]

    return cols, cols_flat
