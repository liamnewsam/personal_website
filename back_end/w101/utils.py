


def inp(values, hidden=[], keys=[], custom=None):
    
    if not keys:
        keys = [str(i) for i in range(1, len(values)+1)]

    
    options = '\n'.join([f"{key}: {value}" for key, value in zip(keys, values)])
    valid = keys+hidden
    while True:
        if not custom:
            print(options)
        else:
            print(custom)
        choice = input("Enter: ")
        if choice in valid:
            return choice
        else:
            print("Invalid entry")