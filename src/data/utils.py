dataset_obj = {}
def register_dataset_obj(name):

    """
    Dataset register
    """

    def decorator(cls):
        dataset_obj[name] = cls
        return cls
    return decorator


def get_dataset(name, **kwargs):

    """
    Dataset getter
    """

    return dataset_obj[name](**kwargs)


