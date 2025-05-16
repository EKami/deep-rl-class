fn argmax[
    dtype: DType, T: Copyable & Movable & Comparable
](vec: List[T]) raises -> Int:
    """
    Returns the index of the maximum element in the DynamicVector.
    If multiple occurrences of the maximum value exist, the index of the
    first occurrence is returned.
    Raises an Error if the vector is empty.
    """
    var n = len(vec)
    if n == 0:
        raise Error("argmax of empty sequence")

    var max_val: T = vec[0]
    var max_idx: Int = 0  # Or use Index type: var max_idx: Index = 0

    for i in range(1, n):  # Start from the second element
        if vec[i] > max_val:
            max_val = vec[i]
            max_idx = i

    return max_idx
