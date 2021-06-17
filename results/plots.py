"""
Author: Danilo de Goede.

plots.py:
This file contains functionality to generate LaTeX code for plotting graphs
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

NUM_FILES = 5
BOARD_REPR_METHODS = ['original', 'symmetric', 'simple', 'spatial_planes', 'spatial_planes_4']
LAYERS_NUMBERS = ['4-layers', '6-layers', '8-layers']
LAYERS_LEGEND = ['4 layers', '6 layer', '8 layers']
BOARD_REPR_METHODS_LEGEND = ['Original', 'Symmetric', 'Simple', 'Binary planes', 'Hybrid']
COLORS = ['red', 'green', 'blue', 'violet', 'orange']


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def get_coords(x_list, y_list, round=False):
    """get_coords([1, 2, 3], [4, 5, 6]) -> str('(1, 4)(2, 5)(3, 6)')"""
    if not round:
        return ''.join([f"({x}, {y})" for x, y in zip(x_list, y_list)])
    else:
        return ''.join([f"({np.around(x, 2)}, {np.around(y, 2)})" for x, y in zip(x_list, y_list)])


def get_coords_std(x_list, y_list, stds):
    return ' '.join([f"({x}, {np.around(y, 2)})+-(0, {std})" for x, y, std in zip(x_list, y_list, stds)])


def get_coords_fill_between(x_list, y_std):
    return ''.join([f"({x}, {np.around(y, 2)})" for x, y in zip(x_list, y_std)])


def add_plot(x_list, y_list, name_legend, color, mark="*", bar=False, stds=None):
    """Data is a list of lists, each sublist has the form [xdata, ydata, name_legend]"""
    color_string = f"""[
        mark={mark},
        color={color},
        dashed
    ]"""
    std_string = f"""+[
        error bars/.cd,
        y dir=both,
        y explicit
    ]"""

    return f"""
\\addplot{color_string if not bar else std_string}
    coordinates {{
        {get_coords(x_list, y_list) if stds is None else get_coords_std(x_list, y_list, stds)}
    }};
    \\addlegendentry{{{name_legend}}}"""


def add_plot_fill_between(x_list, y_list, std_list, path, color):
    """Add a single plot that displays standard deviation."""
    y_std_upper = y_list + std_list
    y_std_lower = y_list - std_list

    return f"""
\\addplot [color={color}] coordinates {{{get_coords(x_list, y_list, round=True)}}};

\\addplot [name path={path}_top, color={color}!70, forget plot] coordinates {{{get_coords_fill_between(x_list, y_std_upper)}}};

\\addplot [name path={path}_down, color={color}!70, forget plot] coordinates {{{get_coords_fill_between(x_list, y_std_lower)}}};

\\addplot [{color}!50,fill opacity=0.5, forget plot] fill between[of={path}_top and {path}_down];
"""


def add_plots(data, bar=False):
    """Add a plot for every line."""
    if not bar:
        return '\n'.join([add_plot(x_list, y_list, name_legend, color, bar=bar) for (x_list, y_list, name_legend), color in zip(data, COLORS)])
    else:
        return '\n'.join([add_plot(x_list, y_list, name_legend, color, bar=bar, stds=stds) for (x_list, y_list, stds, name_legend), color in zip(data, COLORS)])


def add_plots_fill_between(data):
    """Add a plot for every line with fill-between to display the standard deviation."""
    return '\n'.join([add_plot_fill_between(x_list, y_list, std_list, path, color) for (x_list, y_list, std_list, path, color) in data])


def latex_template(arg_dict, nn_architecture=False):
    """Main function for line graphs."""
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
    width={"0.8" if not nn_architecture else "1.05"}\\textwidth,
    height={"0.55" if not nn_architecture else "0.8"}\\textwidth,
    xlabel={{{arg_dict.xlabel}}},
    ylabel={{{arg_dict.ylabel}}},
    {"ylabel near ticks," if nn_architecture else ""}
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed
]

{add_plots(arg_dict.data)}

\\end{{axis}}
\\end{{tikzpicture}}
    """


def latex_template_bar(arg_dict, nn_architecture=False):
    """Main function for bar graphs"""
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
    symbolic x coords={{{','.join(BOARD_REPR_METHODS_LEGEND) if not nn_architecture else ','.join(LAYERS_LEGEND)}}},
    xtick=data,
    ybar,
    ymin=0,
    ymax=20,
    width={"0.7" if not nn_architecture else "1.05"}\\textwidth,
    height={"0.4" if not nn_architecture else "0.8"}\\textwidth,
    {"ylabel near ticks," if nn_architecture else ""}
    xlabel={{{arg_dict.xlabel}}},
    ylabel={{{arg_dict.ylabel}}},
    legend pos=north west,
    ymajorgrids=true,
    grid style=dashed
]

{add_plots(arg_dict.data, bar=True)}

\\end{{axis}}
\\end{{tikzpicture}}
    """

def latex_template_fill_between(arg_dict, horizontal_line_args=None, legend_args=None):
    """Main function for line graphs."""
    text_width = "\\textwidth,"
    return f"""\\begin{{tikzpicture}}
\\begin{{axis}}[
    xlabel={{{arg_dict.xlabel}}},
    ylabel={{{arg_dict.ylabel}}},
    width={"0.8" if legend_args else ""}\\textwidth,
    {"height=0.6" if legend_args else ""}{text_width if legend_args else ""}
    legend pos=north {"west" if legend_args else "east"},
    ymajorgrids=true,
    grid style=dashed,
]

{add_plots_fill_between(arg_dict.data)}
{plot_horizontal_line(*horizontal_line_args) if horizontal_line_args else ""}
{plot_legend(legend_args) if legend_args else ""}

\\end{{axis}}
\\end{{tikzpicture}}
    """


def read_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        return np.array([row for row in reader], dtype=float).T


def parse_rejected_models(path):
    """Given a CSV files containing a column of wins and a column of losses,
    determine the number of model iterations and the number of total iterations"""
    wins, losses = read_csv(path)
    winrates = wins / (wins + losses)
    total_iterations = np.size(winrates)
    num_model_iterations = np.size(winrates[winrates >= 0.5])

    return [num_model_iterations, total_iterations]


def csvs_to_winrates(path):
    """Given a path to a folder containing csv files whose rows contains the
    model iteration, number of wins and the number of losses, return 2 arrays.
    One for the model iterations, and one for the corresponding mean winrates.
    The average winrate is taken over multiple runs. If a subset of the runs
    did not make it to a certain iteration, these runs will not be taken into
    account in the winrate of this iteration (by padding it with np.nan)"""
    winrates = [csv_to_winrates(f"{path}/{i}/random_baseline.csv") for i in range(1, NUM_FILES + 1)]
    max_training_length = max([len(w) for w in winrates])
    winrates = np.array([w + (max_training_length - len(w)) * [np.nan] for w in winrates])

    winrate_stds = np.around(np.nanstd(winrates, axis=0), decimals=2)
    winrates = np.around(np.nanmean(winrates, axis=0), decimals=2)

    model_iterations = np.arange(1, len(winrates) + 1)


    board_repr_name = path.split('/')[-1]
    print(board_repr_name)
    print("==================================================")
    for x, std in zip(model_iterations, winrate_stds):
        print(x, std)
    print("==================================================")

    return model_iterations, winrates


def csv_to_winrates(path):
    """Given a path to a csv file where each row contains the model iteration,
    number of wins and the number of losses, return 2 arrays. One for the model
    iterations, and one for the corresponding winrates."""
    data = read_csv(path)
    model_iters, wins, losses = data
    winrates = np.around(wins / (wins + losses), decimals=2)
    return list(winrates)


def plot_winrates(action_encoding="tile_relative"):
    data = [(*csvs_to_winrates(f"{action_encoding}/regular/{board_repr}"), name) for board_repr, name in zip(BOARD_REPR_METHODS, BOARD_REPR_METHODS_LEGEND)]

    arg_dict = dotdict({
        'xlabel': "Iteration of model",
        'ylabel': "Win rate",

        # Data is a list of tuples, each tuple has the form [xdata, ydata, name_legend, color]
        'data': data
    })
    return latex_template(arg_dict)


def plot_rejections(action_encoding="tile_relative"):
    """Create a bar graph that shows the number of rejections for different board representations."""
    final_data = []
    final_stds = []

    for rep in BOARD_REPR_METHODS:
        data = np.array([parse_rejected_models(f"{action_encoding}/regular/{rep}/{i}/stage3.csv") for i in range(1, NUM_FILES + 1)])
        averages = np.around(np.mean(data, axis=0), decimals=2)
        stds = np.around(np.std(data, axis=0), decimals=2)
        final_data.append(averages)
        final_stds.append(stds)

    final_data = np.array(final_data).T
    final_stds = np.array(final_stds).T

    legend_names = ["Accepted", "Total"]
    final_data = [(BOARD_REPR_METHODS_LEGEND, avg, std, legend) for avg, std, legend in zip(final_data, final_stds, legend_names)]

    arg_dict = dotdict({
        'xlabel': "Board representation",
        'ylabel': "Number of models",

        # Data is a list of tuples, each tuple has the form [xdata, ydata, name_legend, color]
        'data': final_data
    })

    return latex_template_bar(arg_dict)


def plots_policy_value_loss(rep="spatial_planes"):
    """Create a plot for the policy and value loss for the validation of the training infrastructure."""
    data = np.array([read_csv(f"tile_relative/regular/{rep}/{n}/stage2.csv")[:, :100] for n in range(1, NUM_FILES + 1)])
    print(data)
    p_losses, v_losses = data[:, 0, :], data[:, 1, :]
    p_loss, p_loss_std = np.mean(p_losses, axis=0), np.std(p_losses, axis=0)
    v_loss, v_loss_std = np.mean(v_losses, axis=0), np.std(v_losses, axis=0)

    num_epochs_x = np.arange(1, np.size(p_loss) + 1)

    data = [(num_epochs_x, p_loss, p_loss_std, "policy", "red")]
    arg_dict = dotdict({
        'xlabel': "Epoch",
        'ylabel': "Policy loss",
        'data': data
    })
    return_template = latex_template_fill_between(arg_dict)

    data = [(num_epochs_x, v_loss, v_loss_std, "value", "blue")]
    arg_dict = dotdict({
        'xlabel': "Epoch",
        'ylabel': "Value loss",
        'data': data
    })
    return_template += '\n' + latex_template_fill_between(arg_dict)

    return return_template

def plot_rejections_nn_architecture(variant='regular'):
    """Create a bar graph that shows the number of rejections for nn architectures."""
    final_data = []
    final_stds = []

    for num_layers in LAYERS_NUMBERS:
        data = np.array([parse_rejected_models(f"nn_architecture/{variant}/{num_layers}/{i}/stage3.csv") for i in range(1, NUM_FILES + 1)])
        averages = np.around(np.mean(data, axis=0), decimals=2)
        stds = np.around(np.std(data, axis=0), decimals=2)
        final_data.append(averages)
        final_stds.append(stds)

    final_data = np.array(final_data).T
    final_stds = np.array(final_stds).T

    legend_names = ["Accepted", "Total"]
    final_data = [(LAYERS_LEGEND, avg, std, legend) for avg, std, legend in zip(final_data, final_stds, legend_names)]

    arg_dict = dotdict({
        'xlabel': "Number of layers",
        'ylabel': "Number of models",

        # Data is a list of tuples, each tuple has the form [xdata, ydata, name_legend, color]
        'data': final_data
    })

    return latex_template_bar(arg_dict, nn_architecture=True)

def plot_winrates_nn_architecture(variant='regular'):
    data = [(*csvs_to_winrates(f"nn_architecture/{variant}/{num_layers}"), name) for num_layers, name in zip(LAYERS_NUMBERS, LAYERS_LEGEND)]

    arg_dict = dotdict({
        'xlabel': "Iteration of model",
        'ylabel': "Win rate",

        # Data is a list of tuples, each tuple has the form [xdata, ydata, name_legend, color]
        'data': data
    })
    return latex_template(arg_dict, nn_architecture=True)

def plot_horizontal_line(color, xmin, xmax, y):
    return f"\\addplot [color={color}] coordinates {{({xmin}, {y})({xmax}, {y})}};"

def plot_legend(legend_list):
    return f"\\legend{{{', '.join(legend_list)}}}"

def plot_usability_real_life():
    """Plot the execution time of a single move versus the number of MCTS sims."""
    data = np.around(np.array(read_csv(f"usability_real_life/usability_analysis.csv")), 3).T

    num_mcts_sims = np.arange(100, 1501, 100)
    thinking_time_mean = np.array([np.mean(data[data[:, 0] == n], axis=0)[1] for n in num_mcts_sims])
    thinking_time_std = np.array([np.std(data[data[:, 0] == n], axis=0)[1] for n in num_mcts_sims])

    data = [(num_mcts_sims, thinking_time_mean, thinking_time_std, "desired", "red")]
    arg_dict = dotdict({
        'xlabel': "Number of MCTS simulations",
        'ylabel': "Thinking time per move (s)",
        'data': data
    })

    desired_thinking_time = 15
    horizontal_line_args = ["blue", num_mcts_sims[0], num_mcts_sims[-1], desired_thinking_time]
    legend_args = ["Thinking time engine", "Desirable thinking time"]

    return latex_template_fill_between(arg_dict, horizontal_line_args=horizontal_line_args, legend_args=legend_args)

def next_input():
    inp = input("Go to next plot or exit? [n/e]?")

    if inp != 'n':
        exit()


def main():
    print(plots_policy_value_loss())
    next_input()

    print(plot_rejections(action_encoding="absolute_coord"))
    next_input()
    print(plot_winrates(action_encoding="absolute_coord"))
    next_input()

    print(plot_rejections())
    next_input()
    print(plot_winrates())
    next_input()

    print(plot_rejections_nn_architecture())
    next_input()
    print(plot_winrates_nn_architecture())
    next_input()

    print(plot_rejections_nn_architecture(variant='symmetries'))
    next_input()
    print(plot_winrates_nn_architecture(variant='symmetries'))
    next_input()
    print(plot_usability_real_life())
    next_input()

if __name__ == '__main__':
    main()
