import os


class Paths(object):
    """
    Contains paths to directories used in the analysis.

    The user should modify data_dir; everything else should work from there.
    """
    def __init__(self):

        # top level data directory for the analysis
        self.data_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830/'

        self.features_dir = os.path.join(self.data_dir, 'features')
        self.classification_dir = os.path.join(self.data_dir, 'classification')
        self.results_dir = os.path.join(self.data_dir, 'results')

    def make_directories(self):
        """
        Creates the top level data directories.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
