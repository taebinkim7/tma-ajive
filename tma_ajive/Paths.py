import os


class Paths(object):
    """
    Contains paths to directories used in the analysis.

    The user should modify data_dir; everything else should work from there.
    """
    def __init__(self, data_dir=None):

        # top level data directory for the analysis
        if data_dir is None:
            self.data_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830/'
        else:
            self.data_dir = data_dir

        self.images_dir = os.path.join(self.data_dir, 'images')
        self.features_dir = os.path.join(self.data_dir, 'features')
        self.classification_dir = os.path.join(self.data_dir, 'classification')
        self.ajive_dir = os.path.join(self.data_dir, 'ajive')

    def make_directories(self):
        """
        Creates the top level data directories.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.classification_dir, exist_ok=True)
        os.makedirs(self.ajive_dir, exist_ok=True)
