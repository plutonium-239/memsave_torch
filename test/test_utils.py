"""Tests for the memsave_torch.util package (tests the estimate module which needs everything else to be run)"""


def _dont_print(x):
    pass


def test_all():
    """Tests for a simple deep conv model with relus"""
    import shlex
    import subprocess
    from time import sleep

    from memsave_torch.util import collect_results
    from memsave_torch.util.models import prefix_in_pairs

    estimators = ["time", "memory"]

    # improvements can be either speedups or savings based on context
    vjp_improvements = [
        0.9,  # doable
        0.5,  # realistic
        0.1,  # ambitious
        0.01,  # unrealistic
    ]

    # CONV
    # Valid choices for models are in models.conv_model_fns
    models = ["deeprelumodel"]

    models = prefix_in_pairs("memsave_", models)
    batch_size = 64
    input_channels = 3
    input_HW = 224
    num_classes = 1000
    device = "cuda"
    architecture = "conv"
    results_dir = ".test_results/"

    cases = [
        None,
        [  # CONV
            "no_grad_linear_weights",
            "no_grad_linear_bias",
            "no_grad_norm_weights",
            "no_grad_norm_bias",
        ],
    ]

    collector = collect_results.ResultsCollector(
        batch_size,
        input_channels,
        input_HW,
        num_classes,
        device,
        architecture,
        vjp_improvements,
        cases,
        results_dir,
        print=_dont_print,
    )

    for model in models:
        for estimate in estimators:
            collector.clear_file(estimate)
            for case in cases:
                case_str = f"--case {' '.join(case)}" if case is not None else ""
                cmd = (
                    f"python memsave_torch/util/estimate.py --architecture {architecture} --model {model} --estimate {estimate} {case_str} "
                    + f"--device {device} -B {batch_size} -C_in {input_channels} -HW {input_HW} -n_class {num_classes} "
                    + f"--results_dir {results_dir}"
                )
                proc = subprocess.run(shlex.split(cmd), capture_output=True)
                assert proc.stderr in [
                    None,
                    b"",
                ], f"Error in estimate.py: \n{proc.stderr.decode()}"
                sleep(0.25)

            collector.collect_from_file(estimate, model)

    collector.finish()

    import memsave_torch.get_best_results

    memsave_torch.get_best_results.main(results_dir)
