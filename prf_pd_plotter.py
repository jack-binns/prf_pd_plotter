import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

# Global plotting params
# Feel free to import matplotlib style sheet instead
plt.rcParams['axes.linewidth'] = 0.2  # set the value globally
plt.rcParams["font.family"] = "Arial"


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


class PowderFigure:
    """
    This object handles the display and collection of data into a figure
    """

    def __init__(self, root: str = 'root', tag: str = 'tag', theta_lims: tuple = (0, 100)):
        self.root = root
        self.tag = tag

        self.theta_lims = theta_lims
        self.log_scale = False

        self.dset_list = []

    def load_dataset(self, root: str = '', dset_tag: str = 'tag1'):
        # We assume you'll be based in the same root dir, but this can be changed
        if not root:
            root = self.root
        dset = PowderDataset(root=root, tag=dset_tag)
        self.dset_list.append(dset)

    def plot_lebail(self, ticks: bool = True, tick_labels: bool = True, tick_label_lim: int = 40,
                    title: str = '', show=False):

        plt.figure(figsize=cm2inch(8.6, 5.342))  # Sized for double columned articles as cm2inch(8.6,5.342)
        plt.xlim(self.theta_lims)
        plt.ylabel('Intensity / arb. units')
        plt.xlabel(r"2$\theta$ / $^\circ$")

        for dset in self.dset_list:
            print(dset.ticks_df.shape)
            diff_offset = np.max(np.abs(dset.profile_df['diff']))
            ticks_offset = (np.min((dset.profile_df['diff'] - diff_offset))) * 1.1
            plt.plot(dset.profile_df['two_theta'], dset.profile_df['bg_corr_model'], 'red', label='Calculated')
            plt.plot(dset.profile_df['two_theta'], dset.profile_df['bg_corr_obs'], 'k+', label='Observed', markersize=3)
            plt.plot(dset.profile_df['two_theta'], dset.profile_df['diff'] - diff_offset, 'black')
            if ticks:
                dset.ticks_df['y_pos'] = ticks_offset
                plt.plot(dset.ticks_df['two_theta'], dset.ticks_df['y_pos'], '|')
            if tick_labels:
                for k, tick in enumerate(dset.ticks_df[:tick_label_lim].iterrows()):
                    plt.text(dset.ticks_df['two_theta'][k], ticks_offset * 1.25,
                             f"{dset.ticks_df['h'][k]}{dset.ticks_df['k'][k]}{dset.ticks_df['l'][k]}", ha='center',
                             va='center', rotation=90)
        plt.yticks([])  # This gets rid of the y value numbers
        # plt.title(title)
        plt.legend()
        plt.tight_layout(pad=0.5)
        # plt.show()
        file_name = pathlib.Path(f'{self.root}\\{title}_LeBail.png')
        print(f'Saving figure to {file_name}')
        plt.savefig(file_name, format='png', dpi=400)
        if show:
            plt.show()

class PowderDataset:
    """
    This object handles a single data set
    """

    def __init__(self, root: str = 'root', tag: str = 'tag'):
        self.root = root
        self.tag = tag

        self.ticks = []
        self.profile = []

        self.ticks_df = []
        self.profile_df = []

        self.dataset = {}

        self.grok_prf()

    def grok_prf(self):
        with open(list(pathlib.Path(f'{self.root}').glob('*.prf'))[0], 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        # print(lines)

        mode = 'ticks'
        for k, line in enumerate(lines):
            if k == 0:
                continue
            if line != '999' and mode == 'ticks':
                self.ticks.append(line.split())
                # print(line.split())
                continue
            elif line == '999' and mode == 'ticks':
                mode = 'profile'
                continue
            elif line != '999.' and mode == 'profile':
                self.profile.append(line.split())
                # print(line.split())
            elif line == '999.':
                continue
        print(f'I found {len(self.ticks)} ticks')
        print(f'I found {len(self.profile)} profile data points')
        self.profile = np.array(self.profile)

        self.ticks_df = pd.DataFrame(self.ticks)
        self.profile_df = pd.DataFrame(self.profile)

        self.ticks_df.columns = ['h', 'k', 'l', 'mult', 'phase', 'two_theta', 'dummy_x', 'dummy_y', 'dummy_z']
        self.profile_df.columns = ['two_theta', 'obs', 'model', 'unk_1', 'phase', 'unk_2', 'valid', 'unk_3', 'bg']
        self.profile_df = self.profile_df.astype(float)
        self.ticks_df = self.ticks_df.astype({'h': int,
                                              'k': int,
                                              'l': int,
                                              'mult': float,
                                              'two_theta': float})
        self.profile_df['bg_corr_obs'] = self.profile_df['obs'] - self.profile_df['bg']
        self.profile_df['bg_corr_model'] = self.profile_df['model'] - self.profile_df['bg']
        self.profile_df['diff'] = self.profile_df['model'] - self.profile_df['obs']
        self.dataset['ticks'] = self.ticks_df
        self.dataset['profile'] = self.profile_df
        self.dataset['tag'] = self.tag
        return self.dataset


if __name__ == '__main__':
    powfig = PowderFigure(root='.\\test_data\\', tag='test')
    print('<prf_pd_plotter> running tests...')
    tags = ['test']
    for tag in tags:
        powfig.load_dataset(dset_tag='test')
    powfig.theta_lims = (1, 25)
    powfig.plot_lebail(show=True)
