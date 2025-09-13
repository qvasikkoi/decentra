from roboflow import Roboflow

rf = Roboflow(api_key="geMRszxYymNb0PdkRkcz")
project = rf.workspace("decentra-vlprm").project("detect-count-and-visualize-2")
dataset = project.version(1).download("yolov8")
