import os
import glob
import re
import csv

def compile_reports_to_csv(search_directories, output_csv="combined_traffic_results.csv"):
    file_list = []
    
    # Ensure search_directories is a list even if a single string is passed
    if isinstance(search_directories, str):
        search_directories = [search_directories]

    # Recursively search through all provided directories
    for directory in search_directories:
        search_pattern = os.path.join(directory, "**", "report_*.txt")
        found_files = glob.glob(search_pattern, recursive=True)
        file_list.extend(found_files)
        
    # Remove duplicates just in case folder paths overlap
    file_list = list(set(file_list))
    
    if not file_list:
        print("No report files found in the provided directories.")
        return

    print(f"Found {len(file_list)} report files. Processing...")

    header_pattern = re.compile(r"^(.*?):\s*(\d+)\s*(?:veh/h/lane|vph)\s*\|\s*(\d+)%\s*PR\s*\(Over\s*(\d+)\s*Seeds\)", re.MULTILINE)
    
    # Updated to capture all the new granular evaluation metrics from the recent scripts
    metric_patterns = {
        'Equivalent Hourly Throughput (veh/h)': re.compile(r"Equivalent Hourly Throughput:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*veh/h"),
        'Raw Arrivals (veh)': re.compile(r"Raw Arrivals \(.*?Episode\):\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*vehicles"),
        'Mean Speed (m/s)': re.compile(r"Mean Speed:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*m/s"),
        'Mean Travel Time (s)': re.compile(r"Mean Travel Time:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*s"),
        
        'Collision Rate (%)': re.compile(r"Collision Rate:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*%"),
        'Avg Near-Collisions': re.compile(r"Avg Near-Collisions.*?:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*per episode"),
        'Avg TTC Violations': re.compile(r"Avg TTC Violations.*?:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*per episode"),
        'Avg Hard Braking': re.compile(r"Avg Hard Braking:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*per episode"),
        
        'Acceleration Var (m/s^2)': re.compile(r"Acceleration Var:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*m/s\^2"),
        'Mean Abs Jerk (m/s^3)': re.compile(r"Mean Abs Jerk:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*m/s\^3"),
        'Wave Intensity (Var)': re.compile(r"Wave Intensity \(Var\):\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*\(speed variance\)"),
        
        'Raw Interventions (Sum)': re.compile(r"Raw Interventions \(Sum\):\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*steps"),
        'Interventions Per CAV': re.compile(r"Interventions Per CAV:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*steps/CAV"),
        'Duty Cycle (%)': re.compile(r"Duty Cycle.*?:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*%"),
        'Mean Decel Magnitude (m/s^2)': re.compile(r"Mean Decel Magnitude:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*m/s\^2"),
        'Mean Duration (s)': re.compile(r"Mean Duration:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*s"),
        
        'Success Rate (%)': re.compile(r"Success Rate:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*%"),
        'Avg Max Queue (veh)': re.compile(r"Avg Max Queue:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*vehicles"),
        'Absolute Max Queue (veh)': re.compile(r"Absolute Max Queue:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*vehicles"),
        
        'Mean Delay (s)': re.compile(r"Mean Delay:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*s"),
        '90th-Percentile Delay (s)': re.compile(r"90th-Percentile Delay:\s*([0-9\.]+)\s*[^\d\.]+\s*([0-9\.]+)\s*s")
    }

    parsed_data = []

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as file:
            content = file.read()

            header_match = header_pattern.search(content)
            if not header_match:
                print(f"Skipping {os.path.basename(file_path)}: Could not find standard header.")
                continue
            
            model_name = header_match.group(1).strip()
            demand = header_match.group(2)
            pr = header_match.group(3)
            seeds = header_match.group(4)

            row_data = {
                'Model': model_name,
                'Demand (vph/lane)': int(demand),
                'Penetration Rate (%)': int(pr),
                'Seeds Tested': int(seeds)
            }

            for column_name, pattern in metric_patterns.items():
                match = pattern.search(content)
                if match:
                    # Combine Group 1 (Mean) and Group 2 (CI) into a single string with a clean ± symbol
                    row_data[column_name] = f"{match.group(1)} ± {match.group(2)}"
                else:
                    row_data[column_name] = None

            parsed_data.append(row_data)

    # Sort the data logically: By Model -> Demand -> Penetration Rate
    parsed_data.sort(key=lambda x: (x['Model'], x['Demand (vph/lane)'], x['Penetration Rate (%)']))

    # Write to CSV
    if parsed_data:
        csv_headers = list(parsed_data[0].keys())
        
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            for row in parsed_data:
                writer.writerow(row)
                
        print(f"\nSuccessfully compiled {len(parsed_data)} reports into '{output_csv}'!")

if __name__ == "__main__":
    folders_to_search = [
        "./results/MAPPO_Curriculum_seed_0809/models",   
        "./results_dongchen_baseline/0809/models",
        "./results/baselineFinal"
    ]
    
    compile_reports_to_csv(search_directories=folders_to_search, output_csv="combined_traffic_results.csv")