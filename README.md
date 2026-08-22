Name: Justin Weng
Student ID: 1155282333

Requirments (Library Versions):
absl-py==2.5.0
annotated-types==0.7.0
attrs==26.1.0
certifi==2026.6.17
charset-normalizer==3.4.9
click==8.4.2
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
dm-tree==0.1.10
Farama-Notifications==0.0.6
filelock==3.32.0
fonttools==4.63.0
fsspec==2026.6.0
gymnasium==1.2.2
idna==3.18
Jinja2==3.1.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.0
lz4==4.4.5
MarkupSafe==3.0.3
matplotlib==3.10.9
mpmath==1.3.0
msgpack==1.2.1
networkx==3.4.2
numpy==2.2.6
opencv-python==5.0.0.93
ormsgpack==1.12.2
packaging==26.0
pandas==2.3.3
pettingzoo==1.26.1
pillow==12.3.0
protobuf==7.35.1
pyarrow==25.0.0
pydantic==2.13.4
pydantic_core==2.46.4
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.2
PyYAML==6.0.3
ray==2.56.1
referencing==0.37.0
requests==2.34.2
rpds-py==0.30.0
scipy==1.15.3
seaborn==0.13.2
six==1.17.0
sumo-rl==1.4.5
sumolib==1.27.1
sympy==1.14.0
tensorboardX==2.6.5
torch==2.13.0
traci==1.27.1
typing-inspection==0.4.2
typing_extensions==4.16.0
tzdata==2026.3
urllib3==2.7.0
wrapt==2.2.2

Files:
train_"   ": Training code for MARL methods
evaluate_"   ": Evaluation code for methods
record_"   ": Code used to generate clips
csv_Maker: Combines evaluation results into a .csv file (combined_traffic_results)
graphs: Code used to generate graphs

Folders:
SUMO_network: contains files that generate environment in SUMO
"   "_results: folders containing results from evaluation codes
MAPPO_utils: code for MAPPO algorithm
graphs_outputs: graphs produced from graphs code
