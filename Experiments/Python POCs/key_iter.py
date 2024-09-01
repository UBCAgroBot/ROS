from jtop import jtop

with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    if jetson.ok():
        # Print all cpu
        for idx, cpu in enumerate(jetson.cpu['cpu']):
            print("------ CPU{idx} ------".format(idx=idx))
            for key, value in cpu.items():
                print("{key}: {value}".format(key=key, value=value))
        # read aggregate CPU status
        total = jetson.cpu['total']
        print("------ TOTAL ------")
        for key, value in total.items():
            print("{key}: {value}".format(key=key, value=value))