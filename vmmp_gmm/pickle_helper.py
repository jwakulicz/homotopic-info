from pathlib import Path
import pickle
import os

def pickle_save_data(data=None,
                     folder_path=None,
                     file_name=None,
                     print_path=True):
    """
    Exports a python object as a pickle file, given a path and file name. 

    Parameters
    ----------
    data : obj
        A python object to be stored as a pickle file.
    folder_path : str or None
        The desired folder path for exporting the data object. The path can be relative or absolute. 
        Relative means relative to the current directory (i.e. the directory of the script in which this method is being run). 
        Absolute means the full path, including the device root etc. 
        If None, the data object is exported to thecurrent directory.
    file_name : str
        The name of the pickle file to be imported.
    """

    folder_path = folder_path or ''
    path = os.path.join(Path("."), folder_path, file_name)
    outfile = open(path, 'wb')
    pickle.dump(data, outfile)
    outfile.close()
    if print_path:
        print('Data saved to: ' + str(path))


def pickle_load_data(folder_path=None,
                     file_name=None,
                     print_path=True):
    """
    Imports a pickle data object by specifying path and file name. 

    Parameters
    ----------
    folder_path : str or None
        The path to the folder in which the import file is stored. The path can be relative or absolute. 
        Relative means relative to the current directory (i.e. the directory of the script in which this method is being run). 
        Absolute means the full path, including the device root etc. 
        If None, the file is imported from the current directory.
    file_name : str
        The name of the pickle file to be imported.

    Returns
    -------
    obj
        The python object imported from the pickle file.
    """

    folder_path = folder_path or ''
    path = os.path.join(Path("."), folder_path, file_name)
    infile = open(path, 'rb')
    data = pickle.load(infile)
    infile.close()
    if print_path:
        print('Data uploaded from: ' + str(path))
    return data