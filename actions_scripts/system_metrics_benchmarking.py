from jtop import jtop
import csv
import argparse
import os
import time
from collections import deque

# arguments: --duration 1 --file jtop.csv

def calculate_rolling_average(data):
    return {key: sum(values)/len(values) for key, values in data.items()}

def calculate_final_averages(file_path, fieldnames):
    """Calculates the final average for each metric from the CSV file."""
    averages = {field: 0.0 for field in fieldnames}
    count = 0

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            count += 1
            for field in fieldnames:
                averages[field] += float(row[field])

    if count > 0:
        averages = {key: value / count for key, value in averages.items()}

    return averages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    parser.add_argument('--duration', action="store", dest="duration", type=int, default=1)  # 1 minute
    parser.add_argument('--file', action="store", dest="file", default="jtop.csv")
    parser.add_argument('--verbose', action="store_true", dest="verbose", default=True)
    
    args = parser.parse_args()
    
    with jtop() as jetson:
        with open(args.file, 'w', newline='') as csvfile:
            fieldnames = ['cpu_usage', 'memory_usage', 'gpu_usage', 'cpu_temp', 'gpu_temp', 'gpu_mem', 'system_voltage', 'system_current', 'system_power']
            # Initialize csv writer
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Write header
            writer.writeheader()
            # Initialize storage for rolling averages
            rolling_data = {field: deque(maxlen=5) for field in fieldnames}

            count = 0
            while jetson.ok() and count < args.duration * 60:
                cpu_usage = jetson.cpu['total']['user']
                memory_usage = (jetson.memory['RAM']['used'] / jetson.memory['RAM']['tot']) * 100
                gpu_usage = jetson.gpu['gpu']['status']['load']
                cpu_temp = jetson.temperature['cpu']['temp']
                gpu_temp = jetson.temperature['gpu']['temp']
                gpu_mem = int(str(jetson.memory['RAM']['shared'])[:3])
                system_voltage = jetson.power['tot']['volt']
                system_current = jetson.power['tot']['curr']
                system_power = jetson.power['tot']['power']
                
                # Add the current metrics to the rolling data
                rolling_data['cpu_usage'].append(cpu_usage)
                rolling_data['memory_usage'].append(memory_usage)
                rolling_data['gpu_usage'].append(gpu_usage)
                rolling_data['cpu_temp'].append(cpu_temp)
                rolling_data['gpu_temp'].append(gpu_temp)
                rolling_data['gpu_mem'].append(gpu_mem)
                rolling_data['system_voltage'].append(system_voltage)
                rolling_data['system_current'].append(system_current)
                rolling_data['system_power'].append(system_power)

                # Calculate the rolling average every 5 seconds
                if count % 5 == 4:
                    averages = calculate_rolling_average(rolling_data)
                    if args.verbose:
                        print(f"\nRolling Average - Time: {count + 1}s")
                        print(f"CPU Usage: {averages['cpu_usage']:.1f}%")
                        print(f"Memory Usage: {averages['memory_usage']:.1f}%")
                        print(f"GPU Usage: {averages['gpu_usage']:.1f}%")
                        print(f"GPU Memory: {averages['gpu_mem']} MB")
                        print(f"CPU Temperature: {averages['cpu_temp']:.1f}°C")
                        print(f"GPU Temperature: {averages['gpu_temp']:.1f}°C")
                        print(f"System Voltage: {averages['system_voltage']} V")
                        print(f"System Current: {averages['system_current']} mA")
                        print(f"System Power: {averages['system_power']} W")

                    # Write the rolling average to the CSV file
                    writer.writerow(averages)

                count += 1
                time.sleep(1)

    # Calculate final averages after logging is finished
    final_averages = calculate_final_averages(args.file, fieldnames)
    print("\nFinal Averages over the entire interval:")
    for key, value in final_averages.items():
        print(f"{key}: {value:.2f}")