import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Global plotting params
# Feel free to import matplotlib style sheet instead
plt.rcParams['axes.linewidth'] = 0.2  # set the value globally
plt.rcParams["font.family"] = "Arial"


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

        self.grok_prf()

    def grok_prf(self):
        with open(f'{self.root}\\{self.tag}.prf', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        # print(lines)

        mode = 'ticks'
        for k, line in enumerate(lines):
            if k == 0:
                continue
            if line != '999' and mode == 'ticks':
                self.ticks.append(line.split())
                print(line.split())
                continue
            elif line == '999' and mode == 'ticks':
                mode = 'profile'
                continue
            elif line != '999.' and mode == 'profile':
                self.profile.append(line.split())
                print(line.split())
            elif line == '999.':
                continue
        print(f'I found {len(self.ticks)} ticks')
        print(f'I found {len(self.profile)} profile data points')
        print(self.ticks[0])
        print(self.ticks[-1])
        print(self.profile[0])
        print(self.profile[-1])
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
        print(self.profile_df.dtypes)
        print(self.ticks_df.dtypes)


if __name__ == '__main__':
    powfig = PowderFigure(root='.\\test_data\\', tag='test')
    print('<prf_pd_plotter> running tests...')
    powfig.load_dataset(dset_tag='test')
