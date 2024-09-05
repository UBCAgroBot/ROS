from jtop import jtop
import csv
import argparse
import os
import time

# arguments: --duration 1 --file jtop.csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    parser.add_argument('--duration', action="store", dest="duration", type=int, default=1)  # 1 minute
    parser.add_argument('--file', action="store", dest="file", default="jtop.csv")
    parser.add_argument('--verbose', action="store_true", dest="verbose", default=True)
    
    args = parser.parse_args()
    
    with jtop() as jetson:
        with open(args.file, 'w', newline='') as csvfile:
            count = 0
            fieldnames = ['cpu_usage', 'memory_usage', 'gpu_usage', 'cpu_temp', 'gpu_temp', 'gpu_mem', 'system_voltage', 'system_current', 'system_power']
            # Initialize csv writer
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header
            writer.writeheader()
            # Start loop
            os.system("clear")  # start ros2 launch
            while jetson.ok() and count < args.duration * 60:
                cpu_usage = jetson.cpu['total']['system'] + jetson.cpu['total']['user']
                memory_usage = (jetson.memory['RAM']['used'] / jetson.memory['RAM']['tot']) * 100
                gpu_usage = jetson.gpu['ga10b']['status']['load']
                cpu_temp = jetson.temperature['CPU']['temp']
                gpu_temp = jetson.temperature['GPU']['temp']
                gpu_mem = int(str(jetson.memory['RAM']['shared'])[:3])
                system_voltage = jetson.power['tot']['volt']
                system_current = jetson.power['tot']['curr']
                system_power = jetson.power['tot']['power']
                
                if args.verbose:
                    print(f"CPU Usage: {cpu_usage:.1f}%")
                    print(f"Memory Usage: {memory_usage:.1f}%")
                    print(f"GPU Usage: {gpu_usage:.1f}%")
                    print(f"GPU Memory: {gpu_mem} MB")
                    print(f"CPU Temperature: {cpu_temp:.1f}°C")
                    print(f"GPU Temperature: {gpu_temp:.1f}°C")
                    print(f"System Voltage: {system_voltage} V")
                    print(f"System Current: {system_current} mA")
                    print(f"System Power: {system_power} W")
                    
                line = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'gpu_usage': gpu_usage,
                    'cpu_temp': cpu_temp,
                    'gpu_temp': gpu_temp,
                    'gpu_mem': gpu_mem,
                    'system_voltage': system_voltage,
                    'system_current': system_current,
                    'system_power': system_power
                }
                writer.writerow(line)

                count += 1
                time.sleep(1)
    
    print("Logging finished")