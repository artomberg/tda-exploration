import numpy as np

def idx_to_ndarray(idx_stream):
    """
    Converts an IDX binary data stream into an ndarray of the appropriate size.

    Assumptions:    1) the data is in unsigned byte format;
                    2) the number of data items is nonzero.
    """

    num_axis = idx_stream.read(4)[3] - 1 # Number of axis for each item
    tot_items = int.from_bytes(idx_stream.read(4), byteorder='big') # Total number of items

    # Get the dimensions of each item
    dimensions = []
    for i in range(num_axis):
        dimensions.append(int.from_bytes(idx_stream.read(4), byteorder='big'))
        
    # Get the data and check that its size matches with the specified dimensions
    byte_data = bytearray(idx_stream.read())
    tot_size = len(byte_data)
    exp_size = tot_items * int(np.array(dimensions).prod())
    if tot_size != exp_size:
        print("Total size of data does not match expected size!")
        return None
    
    # Load the data into an ndarray, reshape according to the dimensions and return
    ndarray_shape = [tot_items] + dimensions # This will be the shape of our ndarray
    return np.array(byte_data).reshape(ndarray_shape)
    




