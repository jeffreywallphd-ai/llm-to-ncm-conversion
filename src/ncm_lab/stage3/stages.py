"""Owned worker dispatch, composed from the shared production implementation."""


def execute_stage(run, value, bound):
    if value["stage"] == "stage3_evaluate":
        from ncm_lab.stage3.evaluation import evaluate

        return evaluate(run, value, bound)
    from ncm_lab.stage3.data import source, distill, qualify

    return {
        "stage3_source": lambda: source(run, value),
        "stage3_distill": lambda: distill(run, value, bound),
        "stage3_qualify": lambda: qualify(run, value, bound),
    }[value["stage"]]()
