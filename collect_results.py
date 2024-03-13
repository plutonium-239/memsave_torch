import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

strings = {
    "time": ["s", "T", "speed-up", "faster", 'Estimated time speed-up', 'Time Taken (s)'],
    "memory": ["MB", "M", "savings", "memory", 'Estimated memory savings', 'Memory Usage (MB)'],
}


class ResultsCollector:
    def __init__(
        self, batch_size, input_channels, input_HW, num_classes, device, architecture, vjp_improvements, cases
    ) -> None:
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_HW = input_HW
        self.num_classes = num_classes
        self.device = device
        self.architecture = architecture
        self.vjp_improvements = vjp_improvements 
        self.cases = cases
        assert len(cases) == 3, f"len(cases) > 3:\n{cases}"
        self.base_location = f'results/{architecture}-'
        os.makedirs('results/', exist_ok=True)
        self.savings = pd.DataFrame(columns=['model', 'input_vjps', strings['time'][4], strings['memory'][4]])
        self.usage_stats = pd.DataFrame(columns=['model', 'case', strings['time'][5], strings["memory"][5]])
        self.savings.set_index(['model', 'input_vjps'], inplace=True)
        self.usage_stats.set_index(['model', 'case'], inplace=True)

    def collect_from_file(self, estimate, model, case=None):
        with open(f"results/{estimate}-{self.architecture}.txt") as f:
            lines = f.readlines()

        try:
            assert (
                len(lines) == 3
            ), f"More than 3 lines found in results/{estimate}-{self.architecture}.txt"
            outputs = [float(line.strip()) for line in lines]
            for case, out in zip(self.cases, outputs):
                case = 'None' if case is None else ' + '.join(case)
                # print(case, out)
                self.usage_stats.loc[(model, case), strings[estimate][5]] = out

            self.display_run(outputs, estimate, model)
        except AssertionError as e:
            raise e
        except ValueError as e:
            print(
                f'File results/{estimate}-{self.architecture}.txt has unallowed text. Contents: \n{"".join(lines)}'
            )
            raise e
        finally:
            self.clear_file(estimate)


    def clear_file(self, estimate):
        with open(f"results/{estimate}-{self.architecture}.txt", "w") as f:
            f.write("")

    def display_run(self, outputs, estimate, model, print=tqdm.write):
        # print(f"{model} input ({input_channels},{input_HW},{input_HW}) {device}")
        # print('='*78)
        s = f"{model} input ({self.batch_size},{self.input_channels},{self.input_HW},{self.input_HW}) {self.device}"
        tqdm.write(s.center(78, '='))

        print(
            f"{strings[estimate][1]}(forward + grad params): {outputs[0]:.3f}{strings[estimate][0]}"
        )
        print(
            f"{strings[estimate][1]}(forward + grad (x + params)): {outputs[1]:.3f}{strings[estimate][0]}"
        )
        print(
            f"{strings[estimate][1]}(forward + grad (x + params - {self.architecture}_weights)): {outputs[2]:.3f}{strings[estimate][0]}"
        )

        q_conv_weight = outputs[1] - outputs[2]
        ratio = q_conv_weight / outputs[0]
        if estimate == "time":
            print(
                f"{self.architecture.capitalize()} weight VJPs use {100 * ratio:.1f}% of time"
            )
        else:
            print(
                f"Information for {self.architecture} weight VJPs uses {100 * ratio:.1f}% of memory"
            )
        # self.models.loc[model, '']

        tot_improvements = [
            1 - (1 - improvement) * ratio for improvement in self.vjp_improvements
        ]
        for vjp, tot in zip(self.vjp_improvements, tot_improvements):
            print(
                f"Weight VJP {strings[estimate][2]} of {vjp:.2f}x ({1 / vjp:.1f}x {strings[estimate][3]})"
                + f" would lead to total {strings[estimate][2]} of {tot:.2f}x ({1 / tot:.1f}x {strings[estimate][3]})"
            )
            self.savings.loc[(model, vjp), strings[estimate][4]] = f"{1 / tot:.1f}x"
        print("")

    def finish(self):
        time = datetime.now().strftime("%d.%m.%y %H.%M")
        s = f"input ({self.batch_size},{self.input_channels},{self.input_HW},{self.input_HW}) {self.device}"
        savings_path = f'results/savings-{self.architecture}-{self.device}-{time}.csv'
        with open(savings_path, 'w') as f:
            f.write(s+'\n')
        self.savings.to_csv(savings_path, mode='a')

        usage_path = f'results/usage_stats-{self.architecture}-{self.device}-{time}.csv'
        with open(usage_path, 'w') as f:
            f.write(s+'\n')
        self.usage_stats.to_csv(usage_path, mode='a')


def hyperparam_str(args):
    return f"HW={args.input_hw} B={args.batch_size} C_in={args.input_channels}"
