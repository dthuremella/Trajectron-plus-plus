from utils import prediction_output_to_trajectories
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      color='b',
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')
        for sample_num in range(prediction_dict[node].shape[1]):
        # for sample_num in range(prediction_dict[node].shape[1]-1, prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=color,
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')


def visualize_prediction(prediction_output_dict1,
                         prediction_output_dict2,
                         prediction_output_dict3,
                         prediction_output_dict4,
                         batch_error_dict1,
                         batch_error_dict2,
                         batch_error_dict3,
                         batch_error_dict4,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict1, histories_dict1, futures_dict1 = prediction_output_to_trajectories(prediction_output_dict1,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    prediction_dict2, histories_dict2, futures_dict2 = prediction_output_to_trajectories(prediction_output_dict2,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    prediction_dict3, histories_dict3, futures_dict3 = prediction_output_to_trajectories(prediction_output_dict3,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    prediction_dict4, histories_dict4, futures_dict4 = prediction_output_to_trajectories(prediction_output_dict4,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)
    if len(prediction_dict1.keys()) == 0:
        return
    # ts_key = list(prediction_dict.keys())[0]
    for i, ts_key in enumerate(prediction_dict1.keys()):
        if batch_error_dict1[i] < 1.67:
            continue
        fig, ax = plt.subplots()
        node = list(prediction_dict1[ts_key].keys())[0]
        p_d1 = prediction_dict1[ts_key]
        h_d1 = histories_dict1[ts_key]
        f_d1 = futures_dict1[ts_key]
        p_d2 = prediction_dict2[ts_key]
        h_d2 = histories_dict2[ts_key]
        f_d2 = futures_dict2[ts_key]

        p_d3 = prediction_dict3[ts_key]
        h_d3 = histories_dict3[ts_key]
        f_d3 = futures_dict3[ts_key]

        p_d4 = prediction_dict4[ts_key]
        h_d4 = histories_dict4[ts_key]
        f_d4 = futures_dict4[ts_key]                

        if map is not None:
            ax.imshow(map[ts_key][node].as_image(), origin='lower', alpha=0.5)
        plot_trajectories(ax, p_d1, h_d1, f_d1, color='b', *kwargs)
        plot_trajectories(ax, p_d2, h_d2, f_d2, color='r', *kwargs)
        plot_trajectories(ax, p_d3, h_d3, f_d3, color='y', *kwargs)
        plot_trajectories(ax, p_d4, h_d4, f_d4, color='g', *kwargs)
        ax.title.set_text('ADE error of {} with baseline error {}'.format(batch_error_dict1[i],batch_error_dict2[i]))
        plt.show()

def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue' if node.type.name == 'VEHICLE' else 'orange')
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi/10)
                ax.add_artist(ell)
