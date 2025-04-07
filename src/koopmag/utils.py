import matplotlib.pyplot as plt
import numpy as np
from magtense.micromag import MicromagProblem
from matplotlib.lines import Line2D


def gen_s_state(
    res: tuple,
    grid_size: tuple,
    cuda: bool = False,
    show: bool = False,
) -> np.ndarray:
    problem_ini = MicromagProblem(
        res=res,
        grid_L=grid_size,
        m0=1 / np.sqrt(3),
        alpha=4.42e3,
        gamma=0.0,
        cuda=cuda,
    )

    h_ext = np.array([1, 1, 1]) / (4 * np.pi * 1e-7)

    def h_ext_fct(t) -> np.ndarray:
        return np.expand_dims(np.where(t < 1e-9, 1e-9 - t, 0), axis=1) * h_ext

    t_out, M_out, _, _, _, _, _ = problem_ini.run_simulation(
        100e-9, 200, h_ext_fct, 2000
    )
    M_sq_ini = np.squeeze(M_out, axis=2)

    if show:
        plt.clf()
        plt.plot(t_out, np.mean(M_sq_ini[..., 0], axis=1), "rx")
        plt.plot(t_out, np.mean(M_sq_ini[..., 1], axis=1), "gx")
        plt.plot(t_out, np.mean(M_sq_ini[..., 2], axis=1), "bx")
        plt.show()

        plt.clf()
        plt.figure(figsize=(8, 2), dpi=80)
        s_state = np.reshape(M_sq_ini[-1], (res[1], res[0], 3))
        plt.quiver(s_state[..., 0], s_state[..., 1], pivot="mid")
        plt.show()

    return M_sq_ini[-1]


def gen_seq(
    m0_state: np.ndarray,
    res: list,
    grid_size: list,
    h_ext: tuple = (0, 0, 0),
    t_steps: int = 500,
    t_per_step: float = 4e-12,
    cuda: bool = False,
    show: bool = False,
) -> np.ndarray:
    problem = MicromagProblem(
        res=res,
        grid_L=grid_size,
        m0=m0_state,
        alpha=4.42e3,
        gamma=2.211e5,
        A0=1.3e-11,
        Ms=8e5,
        K0=0.0,
        cuda=cuda,
    )

    t_end = t_per_step * t_steps
    h_ext = np.array(h_ext) / 1000 / (4 * np.pi * 1e-7)

    def h_ext_fct(t) -> np.ndarray:
        return np.expand_dims(t > -1, axis=1) * h_ext

    t_out, M_out, _, _, _, _, _ = problem.run_simulation(
        t_end, t_steps, h_ext_fct, 2000
    )
    M_sq = np.squeeze(M_out, axis=2)

    if show:
        plt.plot(t_out, np.mean(M_sq[..., 0], axis=1), "rx")
        plt.plot(t_out, np.mean(M_sq[..., 1], axis=1), "gx")
        plt.plot(t_out, np.mean(M_sq[..., 2], axis=1), "bx")
        plt.show()

    return M_sq


def plot_dynamics(
    t: np.ndarray, M: np.ndarray, field: np.ndarray | None = None
) -> None:
    Mx = np.mean(M[..., 0], axis=1)
    My = np.mean(M[..., 1], axis=1)
    Mz = np.mean(M[..., 2], axis=1)

    _, ax1 = plt.subplots()

    ax1.plot(t, Mx, "rx")
    ax1.plot(t, My, "gx")
    ax1.plot(t, Mz, "bx")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color="r",
            label=r"MagTense $M_x$",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="g",
            label=r"MagTense $M_y$",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="b",
            label=r"MagTense $M_z$",
            linestyle="None",
        ),
    ]
    ax1.legend(handles=legend_elements)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize="14")
    plt.xlabel("Time [s]", fontsize="14")
    plt.ylabel(r"$M_i$" + " [-]", fontsize="14")
    title = "Standard problem 4"
    if field is not None:
        title += f", Field {field}"
    plt.title(title)
    plt.show()
