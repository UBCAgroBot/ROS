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
        
        for idx, cpu in enumerate(jetson.gpu['gpu']):
            print("------ GPU{idx} ------".format(idx=idx))
            for key, value in cpu.items():
                print("{key}: {value}".format(key=key, value=value))
        # read aggregate CPU status
        total = jetson.gpu['total']
        print("------ TOTAL ------")
        for key, value in total.items():
            print("{key}: {value}".format(key=key, value=value))
            
        for idx, cpu in enumerate(jetson.temperature):
            print("------ CPU{idx} ------".format(idx=idx))
            for key, value in cpu.items():
                print("{key}: {value}".format(key=key, value=value))
        # read aggregate CPU status
        total = jetson.temprature['total']
        print("------ TOTAL ------")
        for key, value in total.items():
            print("{key}: {value}".format(key=key, value=value))
        
        print("overview!!!")
        # CPU
        print('*** CPUs ***')
        print(jetson.cpu)
        # CPU
        print('*** Memory ***')
        print(jetson.memory)
        # GPU
        print('*** GPU ***')
        print(jetson.gpu)
        # Engines
        print('*** engine ***')
        print(jetson.engine)
        # nvpmodel
        print('*** NV Power Model ***')
        print(jetson.nvpmodel)
        # jetson_clocks
        print('*** jetson_clocks ***')
        print(jetson.jetson_clocks)
        # Status disk
        print('*** disk ***')
        print(jetson.disk)
        # Status fans
        print('*** fan ***')
        print(jetson.fan)
        # uptime
        print('*** uptime ***')
        print(jetson.uptime)
        # local interfaces
        print('*** local interfaces ***')
        print(jetson.local_interfaces)
        # Temperature
        print('*** temperature ***')
        print(jetson.temperature)
        # Power
        print('*** power ***')
        print(jetson.power)
        
        print(jetson.stats)
        print(jetson.stats.keys())