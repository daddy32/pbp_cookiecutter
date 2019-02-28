import os, errno

def create_dir_if_not_exists(directory):
    """
    checks if directory given by path exists in the filesystem and creates it if it doesn't
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise