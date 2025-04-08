# define several configuration parameters for global experiment
import xml.etree.ElementTree as ET

def parse_config(file_path):
    global safety_para, theta, sigma, k 
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract values from the XML file
        safety_para = root.find("safety_para").text == "true"
        theta = float(root.find("sigma").text)
        sigma = float(root.find("delta").text)
        k = int(root.find("k").text)

    except Exception as e:
        print(f"Error reading configuration: {e}")

